import torch
import torch.nn as nn
import torch.nn.functional as F
from .activations import ACTIVATIONS

# Backbone from https://github.com/samcw/ResNet18-Pytorch/blob/master/ResNet18.ipynb

def rgb_to_xyz(image: torch.Tensor) -> torch.Tensor:
    r"""Converts a RGB image to XYZ.

    Args:
        image (torch.Tensor): RGB Image to be converted to XYZ with shape :math:`(*, 3, H, W)`.

    Returns:
        torch.Tensor: XYZ version of the image with shape :math:`(*, 3, H, W)`.

    Example:
        >>> input = torch.rand(2, 3, 4, 5)
        >>> output = rgb_to_xyz(input)  # 2x3x4x5
    """
    if not isinstance(image, torch.Tensor):
        raise TypeError("Input type is not a torch.Tensor. Got {}".format(type(image)))

    if len(image.shape) < 3 or image.shape[-3] != 3:
        raise ValueError("Input size must have a shape of (*, 3, H, W). Got {}".format(image.shape))

    r: torch.Tensor = image[..., 0, :, :]
    g: torch.Tensor = image[..., 1, :, :]
    b: torch.Tensor = image[..., 2, :, :]

    x: torch.Tensor = 0.412453 * r + 0.357580 * g + 0.180423 * b
    y: torch.Tensor = 0.212671 * r + 0.715160 * g + 0.072169 * b
    z: torch.Tensor = 0.019334 * r + 0.119193 * g + 0.950227 * b

    out: torch.Tensor = torch.stack([x, y, z], -3)

    return out


def rgb_to_lab(image: torch.Tensor) -> torch.Tensor:
    r"""Converts a batch of RGB images to Lab.

    The image data is assumed to be in the range of :math:`[0, 1]`. Lab
    color is computed using the D65 illuminant and Observer 2.

    Args:
        image (torch.Tensor): RGB Image to be converted to Lab with shape :math:`(*, 3, H, W)`.

    Returns:
        torch.Tensor: Lab version of the image with shape :math:`(*, 3, H, W)`.

    Example:
        >>> input = torch.rand(2, 3, 4, 5)
        >>> output = rgb_to_lab(input)  # 2x3x4x5
    """
    if not isinstance(image, torch.Tensor):
        raise TypeError("Input type is not a torch.Tensor. Got {}".format(type(image)))

    if len(image.shape) < 3 or image.shape[-3] != 3:
        raise ValueError("Input size must have a shape of (*, 3, H, W). Got {}".format(image.shape))

    # Convert from Linear RGB to sRGB
    r: torch.Tensor = image[..., 0, :, :]
    g: torch.Tensor = image[..., 1, :, :]
    b: torch.Tensor = image[..., 2, :, :]

    rs: torch.Tensor = torch.where(r > 0.04045, torch.pow(((r + 0.055) / 1.055), 2.4), r / 12.92)
    gs: torch.Tensor = torch.where(g > 0.04045, torch.pow(((g + 0.055) / 1.055), 2.4), g / 12.92)
    bs: torch.Tensor = torch.where(b > 0.04045, torch.pow(((b + 0.055) / 1.055), 2.4), b / 12.92)

    image_s = torch.stack([rs, gs, bs], dim=-3)

    xyz_im: torch.Tensor = rgb_to_xyz(image_s)

    # normalize for D65 white point
    xyz_ref_white = torch.tensor([0.95047, 1.0, 1.08883], device=xyz_im.device, dtype=xyz_im.dtype)[..., :, None, None]
    xyz_normalized = torch.div(xyz_im, xyz_ref_white)

    power = torch.pow(xyz_normalized, 1 / 3)
    scale = 7.787 * xyz_normalized + 4.0 / 29.0
    xyz_int = torch.where(xyz_normalized > 0.008856, power, scale)

    x: torch.Tensor = xyz_int[..., 0, :, :]
    y: torch.Tensor = xyz_int[..., 1, :, :]
    z: torch.Tensor = xyz_int[..., 2, :, :]

    L: torch.Tensor = (116.0 * y) - 16.0
    a: torch.Tensor = 500.0 * (x - y)
    _b: torch.Tensor = 200.0 * (y - z)

    out: torch.Tensor = torch.stack([L, a, _b], dim=-3)

    return out


class ResidualBlock(nn.Module):
    def __init__(self, inchannel, outchannel, stride=1, activation='ReLU'):
        super(ResidualBlock, self).__init__()
        self.activation = activation
        self.left = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(outchannel),
            ACTIVATIONS[self.activation],
            nn.Conv2d(outchannel, outchannel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(outchannel)
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or inchannel != outchannel:
            self.shortcut = nn.Sequential(
                nn.Conv2d(inchannel, outchannel, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(outchannel)
            )
            
    def forward(self, x):
        out = self.left(x)
        out = out + self.shortcut(x)
        out = ACTIVATIONS[self.activation](out)
        
        return out

class ResNet(nn.Module):
    def __init__(self, input_channels, width_mult=1.0, out_stage=3, activation='ReLU'):
        super(ResNet, self).__init__()
        assert out_stage > 0 and out_stage <= 4, 'OutStage must be within [1,4]'
        self.input_channels = input_channels
        self.out_stage = out_stage
        self.channels = [
            int(width_mult*32),  # 0
            int(width_mult*64),  # 1
            int(width_mult*128), # 2
            int(width_mult*256), # 3
            int(width_mult*512)  # 4
        ]
        self.output_channels = self.channels[out_stage]
        self.inchannel = self.channels[0]
        self.conv1 = nn.Sequential(
            nn.BatchNorm2d(input_channels) if self.input_channels == 6 else nn.Identity(),
            nn.Conv2d(input_channels, self.channels[0], kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(self.channels[0]),
            ACTIVATIONS[activation]
        )
        self.layer1 = self.make_layer(ResidualBlock, self.channels[1], 2, stride=2, activation=activation)
        self.layer2 = self.make_layer(ResidualBlock, self.channels[2], 2, stride=2, activation=activation)
        self.layer3 = self.make_layer(ResidualBlock, self.channels[3], 2, stride=2, activation=activation)        
        self.layer4 = self.make_layer(ResidualBlock, self.channels[4], 2, stride=2, activation=activation)
        
    def make_layer(self, block, channels, num_blocks, stride, activation):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.inchannel, channels, stride, activation))
            self.inchannel = channels
        return nn.Sequential(*layers)
    
    def forward(self, x):
        if self.input_channels == 6:
            r, g, b = x.unbind(1)
            x = torch.stack([r-g, r-b, g-b, r, g, b], dim=1)

        x = self.conv1(x)
        for i in range(1, 5):
            layer = getattr(self, "layer{}".format(i))
            x = layer(x)
            if i == self.out_stage:
                return x

        raise IndexError('{} not valid out stage index. Must be within [1,4]'.format(self.out_stage))
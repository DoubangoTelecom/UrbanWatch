import torch, os, random, numpy as np
import torchvision.transforms as transforms
from PIL import Image, ImageOps
from .warp import ShapeTransform
from .rgb2lxy import Rgb2LXY

class KlassDataset(torch.utils.data.Dataset):
    def __init__(self, cfg, target):
        assert target in ['train', 'val', 'calib'], f'Target ({target}) must be train or val or calib'
        self.cfg = cfg
        self.target = target
        self.transform = ResizeNormalize(cfg)
        # geometric transformations
        self.warp = ShapeTransform(
            **cfg.augment.warp._asdict(),
        )
        
        # read classes
        with open(os.path.join(cfg.datasets.train, '..', "classes.txt"), "r") as f:
            self.class_names = f.read().splitlines()
            print(f'[{self.target}] Number of classes: {len(self.class_names)}')
        
        # read images list
        self.images_folder = cfg.datasets._asdict()[self.target]
        list_file_name = "dataset.txt" if (target == 'calib' and os.path.exists(os.path.join(self.images_folder, "dataset.txt"))) else "list.txt"
        with open(os.path.join(self.images_folder, list_file_name), "r") as f:
            self.images_files = f.read().splitlines()
            print(f'[{self.target}] Number of files: {len(self.images_files)}')
            
        # helper functions
        self.to_ndarray = lambda x: np.array(x) if not isinstance(x, np.ndarray) else x
        self.to_pillow = lambda x: Image.fromarray(x, 'RGB') if isinstance(x, np.ndarray) else x
        self.to_lxy = Rgb2LXY() if self.cfg.model.chroma == 'lxy' else None
            
    def __len__(self):
        return len(self.images_files)
    
    def __getitem__(self, index, **kwargs):
        file_name = self.images_files[index] # format = '#class#_#id#_name.jpg'
        label_id = int(file_name.split('_')[0])
        file_path = os.path.join(self.images_folder, file_name)
        image = Image.open(file_path).convert('RGB')
        
        # Augment
        if self.cfg.augment.enabled:
            image = self._augment(self.to_ndarray(image))
        
        # Convert to LXY if needed
        if self.cfg.model.chroma == 'lxy':
            image = self.to_lxy.process(self.to_ndarray(image))    
        
        return self.transform(self.to_pillow(image)), torch.tensor(label_id)
        
    def _augment(self, img :np.ndarray):

        # Geometry
        img = self.warp(img)

        # Texture
        from imgaug import augmenters as iaa
        sequence = []
        is_active_fn = lambda: (random.randint(0, 1) == 0) # pick 1/2 only, otherwise tooo slow
        is_enabled = lambda x: (getattr(self.cfg.augment.texture, x)[0] > 0 if isinstance(getattr(self.cfg.augment.texture, x), (list, tuple)) else getattr(self.cfg.augment.texture, x))
        if is_active_fn() and is_enabled('average_blur'):
            sequence.append(iaa.AverageBlur(k=tuple(self.cfg.augment.texture.average_blur)))  
        if is_active_fn() and is_enabled('gaussian_blur'):
            sequence.append(iaa.GaussianBlur(sigma=tuple(self.cfg.augment.texture.gaussian_blur)))
        if is_active_fn() and is_enabled('motion_blur'):
            sequence.append(iaa.MotionBlur(k=self.cfg.augment.texture.motion_blur[0], angle=self.cfg.augment.texture.motion_blur[1:]))
        if is_active_fn() and is_enabled('multiply'):
            sequence.append(iaa.Multiply(mul=tuple(self.cfg.augment.texture.multiply), per_channel=random.choice([False, self.cfg.augment.texture.per_channel])))
        if is_active_fn() and is_enabled('multiply_hue'):
            sequence.append(iaa.MultiplyHue(mul=tuple(self.cfg.augment.texture.multiply_hue)))
        if is_active_fn() and is_enabled('multiply_saturation'):
            sequence.append(iaa.MultiplySaturation(mul=tuple(self.cfg.augment.texture.multiply_saturation)))
        if is_active_fn() and is_enabled('gamma_contrast'):
            sequence.append(iaa.GammaContrast(gamma=tuple(self.cfg.augment.texture.gamma_contrast), per_channel=random.choice([False, self.cfg.augment.texture.per_channel])))
        if is_active_fn() and is_enabled('sigmoid_contrast'):
            sequence.append(iaa.SigmoidContrast(gain=tuple(self.cfg.augment.texture.sigmoid_contrast[:2]), cutoff=tuple(self.cfg.augment.texture.sigmoid_contrast[2:]), per_channel=random.choice([False, self.cfg.augment.texture.per_channel])))
        if is_active_fn() and is_enabled('log_contrast'):
            sequence.append(iaa.LogContrast(gain=tuple(self.cfg.augment.texture.log_contrast), per_channel=random.choice([False, self.cfg.augment.texture.per_channel])))
        if is_active_fn() and is_enabled('linear_contrast'):
            sequence.append(iaa.LinearContrast(alpha=tuple(self.cfg.augment.texture.linear_contrast), per_channel=random.choice([False, self.cfg.augment.texture.per_channel])))
        if is_active_fn() and is_enabled('emboss'):
            sequence.append(iaa.Emboss(alpha=tuple(self.cfg.augment.texture.emboss[:2]), strength=tuple(self.cfg.augment.texture.emboss[2:])))
        if is_active_fn() and is_enabled('additive_gaussian_noise'):
            sequence.append(iaa.AdditiveGaussianNoise(scale=self.cfg.augment.texture.additive_gaussian_noise, per_channel=random.choice([False, self.cfg.augment.texture.per_channel])))
        if is_active_fn() and is_enabled('salt'):
            sequence.append(iaa.Salt(p=self.cfg.augment.texture.salt, per_channel=random.choice([False, self.cfg.augment.texture.per_channel])))
        if is_active_fn() and is_enabled('pepper'):
            sequence.append(iaa.Pepper(p=self.cfg.augment.texture.pepper, per_channel=random.choice([False, self.cfg.augment.texture.per_channel])))
        
        # Change to grayscale (probability)
        if random.random() < self.cfg.augment.texture.grayscale:
            img = self.to_ndarray(
                self.to_pillow(img).convert('L').convert('RGB')
            )

        # Apply transformation
        if len(sequence) > 0:
            transforms = iaa.Sequential(sequence, random_order=True)
            img = transforms(images=[img])[0]

        return img
        
class ResizeNormalize(object):

    def __init__(self, opt, interpolation=Image.BILINEAR):
        self.keep_aspect_ratio = opt.model.imgKar
        self.target_size = (opt.model.imgW, opt.model.imgH)
        self.mean = torch.from_numpy(np.array(opt.model.normalize[0], dtype=np.float32).reshape(3, 1, 1) / 255.0)
        self.std_inv = 1.0 / torch.from_numpy(np.array(opt.model.normalize[1], dtype=np.float32).reshape(3, 1, 1) / 255.0)
        self.interpolation = interpolation
        self.toTensor = transforms.ToTensor()

    def __call__(self, image: Image):
        if self.keep_aspect_ratio:
            tmp = ImageOps.contain(image, self.target_size, self.interpolation)
            img = Image.new(tmp.mode, self.target_size, 0)
            img.paste(tmp, (0, 0))
        else:
            img = image.resize(self.target_size, self.interpolation)
        # next code same as ((x - 127.5) / 127.5)
        img = self.toTensor(img) # [0-255] -> [0-1]
        img.sub_(self.mean).mul_(self.std_inv)
        return img
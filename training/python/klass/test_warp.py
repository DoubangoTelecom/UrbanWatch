import numpy as np, cv2, math, random, os
from typing import Dict, Optional, Tuple
from PIL import Image, ImageOps

def get_flip_matrix(prob=0.5):
    F = np.eye(3)
    if random.random() < prob:
        F[0, 0] = -1
    return F


def get_perspective_matrix(perspective, width, height):
    """

    :param perspective:
    :return:
    """
    if perspective == 0:
        return np.eye(3)
    else:
        assert perspective >= 0.0 and perspective <= 0.5, f'perspective({perspective}) must be within [0, 0.5]'
        r = random.choice([1.0, -1.0])
        __offx = lambda: width*random.uniform(0.0,perspective) * r
        __offy = lambda: height*random.uniform(0.0,perspective) * r
                
        src = np.float32([
                [0, 0],
                [width - 1, 0],
                [width - 1, height - 1],
                [0, height - 1]])

        offsets = np.float32([[__offx(), __offy()],
                                [__offx(), __offy()],
                                [__offx(), __offy()],
                                [__offx(), __offy()]])

        dst = (src + offsets).astype(np.float32)
        
        H = cv2.getPerspectiveTransform(src, dst)
        
        return H


def get_2d_rotation_matrix(center, degree=0.0):
    """

    :param degree:
    :return:
    """
    R = np.eye(3)
    if degree != 0:
        a = random.uniform(-degree, degree)
        R[:2] = cv2.getRotationMatrix2D(angle=a, center=center, scale=1)
    return R

def get_3d_rotation_matrix(width, height, degree):
    ''' https://github.com/eborboihuc/rotate_3d/blob/master/image_transformer.py '''
    theta, phi, gamma = [math.radians(random.uniform(-degree, degree)) for _ in range(3)]
    w, h = width, height
    d = np.sqrt(w ** 2 + h ** 2)
    focal = f = d / (2 * np.sin(gamma) if np.sin(gamma) != 0 else 1)
    dz = focal

    # Projection 2D -> 3D matrix
    A1 = np.array([[1, 0, -w / 2],
                   [0, 1, -h / 2],
                   [0, 0, 1],
                   [0, 0, 1]])

    # Rotation matrices around the X, Y, and Z axis
    RX = np.array([[1, 0, 0, 0],
                   [0, np.cos(theta), -np.sin(theta), 0],
                   [0, np.sin(theta), np.cos(theta), 0],
                   [0, 0, 0, 1]])

    RY = np.array([[np.cos(phi), 0, -np.sin(phi), 0],
                   [0, 1, 0, 0],
                   [np.sin(phi), 0, np.cos(phi), 0],
                   [0, 0, 0, 1]])

    RZ = np.array([[np.cos(gamma), -np.sin(gamma), 0, 0],
                   [np.sin(gamma), np.cos(gamma), 0, 0],
                   [0, 0, 1, 0],
                   [0, 0, 0, 1]])

    # Composed rotation matrix with (RX, RY, RZ)
    R = np.dot(np.dot(RX, RY), RZ)

    # Translation matrix
    T = np.array([  [1, 0, 0, 0],
                    [0, 1, 0, 0],
                    [0, 0, 1, dz],
                    [0, 0, 0, 1]])

    # Projection 3D -> 2D matrix
    A2 = np.array([ [f, 0, w/2, 0],
                    [0, f, h/2, 0],
                    [0, 0, 1, 0]])

    # Final transformation matrix
    return np.dot(A2, np.dot(T, np.dot(R, A1)))


def get_scale_matrix(ratio=(1, 1)):
    """

    :param ratio:
    """
    Scl = np.eye(3)
    scale = random.uniform(*ratio)
    Scl[0, 0] *= scale
    Scl[1, 1] *= scale
    return Scl


def get_stretch_matrix(width_ratio=(1, 1), height_ratio=(1, 1)):
    """

    :param width_ratio:
    :param height_ratio:
    """
    Str = np.eye(3)
    Str[0, 0] *= random.uniform(*width_ratio)
    Str[1, 1] *= random.uniform(*height_ratio)
    return Str


def get_shear_matrix(degree=(0, 0)):
    """

    :param degree:
    :return:
    """
    assert len(degree) == 2, f'{degree} must be list of 2'
    Sh = np.eye(3)
    if degree[0] != 0:
        Sh[0, 1] = math.tan(
            random.uniform(-degree[0], degree[0]) * math.pi / 180
        )  # x shear (deg)
    if degree[1] != 0:
        Sh[1, 0] = math.tan(
            random.uniform(-degree[1], degree[1]) * math.pi / 180
        )  # y shear (deg)
    return Sh


def get_translate_matrix(translate, width, height):
    """

    :param translate:
    :return:
    """
    T = np.eye(3)
    T[0, 2] = random.uniform(0.5 - translate, 0.5 + translate) * width  # x translation
    T[1, 2] = random.uniform(0.5 - translate, 0.5 + translate) * height  # y translation
    return T

def get_minimum_dst_shape(
    src_shape: Tuple[int, int],
    dst_shape: Tuple[int, int],
    divisible: Optional[int] = None,
) -> Tuple[int, int]:
    """Calculate minimum dst shape"""
    src_w, src_h = src_shape
    dst_w, dst_h = dst_shape

    if src_w / src_h < dst_w / dst_h:
        ratio = dst_h / src_h
    else:
        ratio = dst_w / src_w

    dst_w = int(ratio * src_w)
    dst_h = int(ratio * src_h)

    if divisible and divisible > 0:
        dst_w = max(divisible, int((dst_w + divisible - 1) // divisible * divisible))
        dst_h = max(divisible, int((dst_h + divisible - 1) // divisible * divisible))
    return dst_w, dst_h

def translate(img, translate, border_reflect):
    height, width = img.shape[:2]
    return cv2.copyMakeBorder(
        img,
        top=int(random.uniform(0, translate)*height),
        bottom=int(random.uniform(0, translate)*height),
        left=int(random.uniform(0, translate)*width),
        right=int(random.uniform(0, translate)*width),
        borderType=cv2.BORDER_REFLECT if border_reflect else cv2.BORDER_CONSTANT,
        value=(0, 0, 0)
    )

def crop(img, ratio):
    height, width = img.shape[:2]
    return img[
        int(random.uniform(0, ratio)*height):height-int(random.uniform(0, ratio)*height),
        int(random.uniform(0, ratio)*width):width-int(random.uniform(0, ratio)*width),
        ...
    ]

class ShapeTransform:
    """Shape transforms including resize, random perspective, random scale,
    random stretch, random rotation, random shear, random translate,
    and random flip.

    Args:
        perspective: Random perspective factor.
        scale: Random scale ratio.
        stretch: Width and height stretch ratio range.
        rotation: Random rotate degree.
        shear: Random shear degree.
        translate: Random translate ratio.
        flip: Random flip probability.
    """

    def __init__(
        self,
        perspective: float = 0.0,
        scale: Tuple[int, int] = (1, 1),
        stretch: Tuple = ((1, 1), (1, 1)),
        rotation: float = 0.0,
        shear: float = (0, 0),
        translate: float = 0.0,
        flip: float = 0.0,
        crop: float = 0.0,
        border_reflect: bool = False,
        **kwargs
    ):
        self.perspective = perspective
        self.scale_ratio = scale
        self.stretch_ratio = stretch
        self.rotation_degree = rotation
        self.shear_degree = shear
        self.flip_prob = flip
        self.translate_ratio = translate
        self.crop_ratio = crop
        self.border_reflect = border_reflect

    def __call__(self, raw_img):
        height, width = raw_img.shape[:2]

        M = np.eye(3)

        P = get_perspective_matrix(self.perspective, width, height)
        M = P @ M

        Scl = get_scale_matrix(self.scale_ratio)
        M = Scl @ M

        Str = get_stretch_matrix(*self.stretch_ratio)
        M = Str @ M

        R = get_2d_rotation_matrix((width*0.5, height*0.5), self.rotation_degree)
        #R = get_3d_rotation_matrix(width, height, self.rotation_degree)
        M = R @ M

        Sh = get_shear_matrix(self.shear_degree)
        M = Sh @ M

        F = get_flip_matrix(self.flip_prob)
        M = F @ M

        #T = get_translate_matrix(self.translate_ratio, width, height)
        #M = T @ M

        img = self._perspective_warp(raw_img, M)
        img = translate(img, self.translate_ratio, self.border_reflect)
        img = crop(img, self.crop_ratio)
        return img

    # https://stackoverflow.com/a/59741739
    def _perspective_warp(self, image: np.ndarray, transform: np.ndarray) -> np.ndarray:
        h, w = image.shape[:2]
        corners_bef = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
        corners_aft = cv2.perspectiveTransform(corners_bef, transform)
        xmin = math.floor(corners_aft[:, 0, 0].min())
        ymin = math.floor(corners_aft[:, 0, 1].min())
        xmax = math.ceil(corners_aft[:, 0, 0].max())
        ymax = math.ceil(corners_aft[:, 0, 1].max())
        translate = np.eye(3)
        translate[0, 2] = -xmin
        translate[1, 2] = -ymin
        corrected_transform = np.matmul(translate, transform)
        return cv2.warpPerspective(image, corrected_transform, (math.ceil(xmax - xmin), math.ceil(ymax - ymin)), borderValue=0,  borderMode=cv2.BORDER_CONSTANT)


def _augment(opt, img):
    
    from imgaug import augmenters as iaa        
    sequence = []
    activate_fn = lambda: True # pick 1/4 only, otherwise tooo slow
    if activate_fn():
        sequence.append(iaa.GaussianBlur(sigma=tuple(opt.gaussian_blur)))
    if activate_fn():
        sequence.append(iaa.Multiply(mul=tuple(opt.multiply), per_channel=random.choice([True, False])))
    if activate_fn():
        sequence.append(iaa.MultiplyHue(mul=tuple(opt.multiply_hue)))
    if activate_fn():
        sequence.append(iaa.MultiplySaturation(mul=tuple(opt.multiply_saturation)))
    if activate_fn():
        sequence.append(iaa.GammaContrast(gamma=tuple(opt.gamma_contrast), per_channel=random.choice([True, False])))
    if activate_fn():
        sequence.append(iaa.AdditiveGaussianNoise(scale=opt.additive_gaussian_noise, per_channel=random.choice([True, False])))

    # Apply transformation
    if len(sequence) > 0:
        transforms = iaa.Sequential(sequence, random_order=True)
        img = transforms(images=[img])[0]

    # Randomly inverse
    #if random.randint(0, 4) == 0:
    #    img = 255 - img

    return img

def resize(image:np.ndarray, target_size=(128,128), keep_aspect_ratio=False):
    if keep_aspect_ratio:
        tmp = ImageOps.contain(Image.fromarray(image, 'RGB'), target_size)
        img = Image.new(tmp.mode, target_size, 0)
        img.paste(tmp, (0, 0))
    else:
        img = Image.fromarray(image, 'RGB').resize(target_size)

    return np.array(img)

FOLDER_PATH = 'C:/Users/dmi83/Downloads/images'
IMAGE_EXTs = ['tif','png','jpg','jpeg','bmp'] 
list_images = lambda dir: np.array([os.path.join(dir, file) for file in os.listdir(dir) if file.split('.')[-1].lower() in IMAGE_EXTs or os.path.isdir(os.path.join(dir, file))])

if __name__ == '__main__':
    images = list_images(FOLDER_PATH)
    assert len(images) > 0, 'Folder is empty'
    transform = ShapeTransform(
        perspective=0.25,
        scale=(0.8, 1.2),
        stretch=((0.8, 1.2), (0.8, 1.2)),
        rotation=20.0,
        shear=(10.0, 10.0),
        translate=0.2,
        flip=0.0,
        crop=0.2
    )
    opt = type('obj', (object,), {
        'gaussian_blur': [0.0, 2.0],
        'multiply': [0.5, 1.2],
        'multiply_hue': [0.5, 1.2],
        'multiply_saturation': [0.5, 1.2],
        'gamma_contrast': [0.5, 2.0],
        'additive_gaussian_noise': 5.0,
        'salt': 0.02,
        'pepper': 0.02,
    })

    for path in images:
        print(f'Processing {path}...')
        image_orig = cv2.imread(path, cv2.IMREAD_COLOR_RGB)
        image_trans = transform(_augment(opt, image_orig))
        cv2.imshow(os.path.basename(path), 
            cv2.cvtColor(np.concatenate((resize(image_orig, (128, 128)), resize(image_trans, (128, 128))), axis=1), cv2.COLOR_RGB2BGR)
        )
        cv2.waitKey(0)
        cv2.destroyWindow(os.path.basename(path))

    print('!! DONE !!')
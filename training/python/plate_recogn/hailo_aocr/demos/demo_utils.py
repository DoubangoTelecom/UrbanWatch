import os, numpy as np
from PIL import Image, ImageOps

def get_image_list(path, image_ext = [".jpg", ".jpeg", ".webp", ".bmp", ".png"]):
    image_names = []
    for maindir, subdir, file_name_list in os.walk(path):
        for filename in file_name_list:
            apath = os.path.join(maindir, filename)
            ext = os.path.splitext(apath)[1]
            if ext in image_ext:
                image_names.append(apath)
    return image_names

def preprocess_image(image :Image, cfg) -> np.ndarray:
    target_size = (cfg.model.imgW, cfg.model.imgH)
    if cfg.model.padding:
        tmp = ImageOps.contain(image, target_size, Image.BILINEAR)
        img = Image.new(tmp.mode, target_size, 0)
        img.paste(tmp, (0, 0))
    else:
        img = image.resize(target_size, Image.BILINEAR)    
    
    mean = np.array(cfg.model.normalize[0], dtype=np.float32).reshape(1, 1, 3)
    std = np.array(cfg.model.normalize[1], dtype=np.float32).reshape(1, 1, 3)
    return ((img.astype(np.float32) - mean) / std).astype(np.float32)

def load_image(path: str, cfg, preprocess=True):
    img = Image.open(path).convert('L' if cfg.model.grayscale else 'RGB')
    if preprocess:
        img = preprocess_image(img, cfg)
    return img

def pred_decode(alphabet :list, codes :np.array, ctc :bool):
    assert(isinstance(alphabet, list))
    assert(len(codes.shape) == 1)
    char_list = []
    for i in range(len(codes)):
        if codes[i] != 0 and (not (i > 0 and codes[i - 1] == codes[i]) or not ctc):
            char_list.append(alphabet[codes[i]])

    return ''.join(char_list)

def grid2coords(grid, width, height):
    assert width==height, f'Next code except width({width})==height({height})'
    # https://docs.pytorch.org/docs/0.3.1/nn.html#torch.nn.functional.grid_sample
    # https://github.com/open-mmlab/mmcv/blob/90d83c94cfb967ef162c449faf559616f31f28c2/mmcv/ops/point_sample.py#L12
    # grid has values in the range of [-1, 1]
    # On C++ code we can use SubMul function
    scale = (width * 0.5)
    shift = (1.0 / width) - 1.0
    return (grid - shift) * scale

def theta2warp(theta, W, H):
    """https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.affine_grid.html"""
    theta = np.concatenate([theta, np.array([[0, 0, 1]])], axis=0)
    N = [
        [2.0/W, 0.0,   -1.0],
        [0.0,   2.0/H, -1.0],
        [0.0,   0.0,    1.0],
    ]
    N_inv = [
        [W*0.5, 0.0,   W*0.5],
        [0.0,   H*0.5, H*0.5],
        [0.0,   0.0,    1.0],
    ]    
    theta_inv = np.linalg.inv(theta)
    M = N_inv @ theta_inv @ N
    
    return M[:2, :]
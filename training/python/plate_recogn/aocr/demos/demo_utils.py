import os, numpy as np, cv2
from PIL import Image

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
    # Using OpenCV (as the pytorch code) instead of Pillow to ease comparison  
    #img = np.array(image.resize(cfg.data.val.input_size, Image.BILINEAR))
    img = cv2.resize(np.array(image), dsize=(cfg.model.imgW, cfg.model.imgH), interpolation=cv2.INTER_LINEAR)
    return ((img.astype(np.float32) - 127.5) / 127.5).astype(np.float32)

def load_image(path: str, cfg, preprocess=True):
    img = Image.open(path).convert('L')
    if preprocess:
        img = preprocess_image(img, cfg)
    return img

def ctc_decode(alphabet :list, codes :np.array):
    assert(isinstance(alphabet, list))
    assert(len(codes.shape) == 1)
    char_list = []
    for i in range(len(codes)):
        if codes[i] != 0 and (not (i > 0 and codes[i - 1] == codes[i])):  # removing repeated characters and blank.
            char_list.append(alphabet[codes[i]])

    return ''.join(char_list)

def grid2coords(grid, width, height):
    assert width==height, 'Next code except width({})==height({})'.format(width, height)
    # https://docs.pytorch.org/docs/0.3.1/nn.html#torch.nn.functional.grid_sample
    # grid has values in the range of [-1, 1]
    # On C++ code we can use SubMul function
    factor = (width * 0.5) - 1.0
    return np.floor((grid + 1.0) * factor).astype(np.float32)

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
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
    # Resize   
    if cfg.model.imgKar:
        tmp = ImageOps.contain(image, (cfg.model.imgW, cfg.model.imgH), Image.BILINEAR)
        img = Image.new(tmp.mode, (cfg.model.imgW, cfg.model.imgH), 0)
        img.paste(tmp, (0, 0))
    else:
        img = image.resize((cfg.model.imgW, cfg.model.imgH), Image.BILINEAR)

    # Normalization
    mean = np.array(cfg.model.normalize[0], dtype=float).reshape([1, 1, 3])
    std = np.array(cfg.model.normalize[1], dtype=float).reshape([1, 1, 3])
    img = ((np.array(img).astype(np.float32) - mean) / std).astype(np.float32)

    return img

def load_image_then_preprocess(path: str, cfg, channel_first :bool=True):
    img = Image.open(path).convert(
        {'rgb':'RGB', 'gray': 'L'}[cfg.model.chroma]
    )
    image = preprocess_image(img, cfg)
    if channel_first:
        image = image.transpose((2, 0, 1)) # nhwc ->  nchw
    return image[None,...]
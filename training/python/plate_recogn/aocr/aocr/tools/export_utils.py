import numpy as np
from PIL import Image, ImageOps

def load_image_then_preprocess(path: str, cfg, channel_first :bool=True):
    target_size = (cfg.model.imgW, cfg.model.imgH)
    image = Image.open(path).convert('RGB')
    
    # Resize
    if cfg.model.padding:
        tmp = ImageOps.contain(image, target_size)
        img = Image.new(tmp.mode, target_size, 0)
        img.paste(tmp, (0, 0))
    else:
        img = image.resize(target_size)
        
    # Grayscale or RGB
    if cfg.model.grayscale:
        img = img.convert('L')
    
    # Normalization
    
    mean = np.array(cfg.model.normalize[0], dtype=float).reshape([1, 1, 3])
    std = np.array(cfg.model.normalize[1], dtype=float).reshape([1, 1, 3])
    img = ((np.array(img).astype(np.float32) - mean) / std).astype(np.float32)
    
    # Transpose
    if channel_first:
        img = img.transpose((2, 0, 1)) # nhwc ->  nchw

    return img[None,...]
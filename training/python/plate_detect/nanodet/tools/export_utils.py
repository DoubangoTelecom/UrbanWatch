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
    #img = np.array(image.resize(cfg.data.val.input_size, Image.NEAREST))
    img = cv2.resize(np.array(image), dsize=cfg.data.val.input_size, interpolation=cv2.INTER_LINEAR)
    mean_std = cfg.data.val.pipeline.normalize
    mean = np.array(mean_std[0], dtype=np.float32).reshape(1, 1, 3) / 255.0
    std = np.array(mean_std[1], dtype=np.float32).reshape(1, 1, 3) / 255.0
    return ((img.astype(np.float32)/255.0 - mean) / std).astype(np.float32)

def load_image_then_preprocess(path: str, cfg):
    img = Image.open(path).convert('RGB')
    image = preprocess_image(img, cfg)
    image = image.transpose((2, 0, 1))[None,...]
    return image
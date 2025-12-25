import os, numpy as np, torch, cv2
from PIL import Image
from nanodet.data.batch_process import stack_batch_img
from nanodet.data.collate import naive_collate

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
    img = cv2.resize(np.array(image), dsize=cfg.data.val.input_size, interpolation=cv2.INTER_LINEAR)
    mean_std = cfg.data.val.pipeline.normalize
    mean = np.array(mean_std[0], dtype=np.float32).reshape(1, 1, 3) / 255.0
    std = np.array(mean_std[1], dtype=np.float32).reshape(1, 1, 3) / 255.0
    return ((img.astype(np.float32)/255.0 - mean) / std).astype(np.float32)

def load_image(path: str, cfg, preprocess=True):
    img = Image.open(path).convert('RGB')
    image = preprocess_image(img, cfg)
    image = image.transpose((2, 0, 1))
    input_img = image if preprocess else np.array(img.resize(cfg.data.val.input_size, Image.BILINEAR))
    img_info = {
        "id": 0,
        "file_name": os.path.basename(path),
        "width": img.size[0],
        "height": img.size[1]
    }
    meta = dict(
        img_info=img_info, 
        raw_img=np.array(img), 
        numpy_img=input_img,
        warp_matrix=[
            [cfg.data.val.input_size[0]/img.size[0], 0, 0],
            [0, cfg.data.val.input_size[1]/img.size[1], 0],
            [0, 0, 1]
        ]
    )
    meta["img"] = torch.from_numpy(image)
    meta = naive_collate([meta]) # is a list
    meta["img"] = stack_batch_img(meta["img"], divisible=32)
    return meta

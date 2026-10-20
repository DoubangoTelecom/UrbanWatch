import numpy as np, os, cv2

FOLDER_IN_PATH = 'E:/Projects/ocr_datasets/alpr-vcr-untouched/train/white'
FOLDER_OUT_PATH = 'E:/Projects/Garbage/vcr'

IMAGE_EXTs = ['tif','png','jpg','jpeg','bmp'] 
list_images = lambda dir: np.array([os.path.join(dir, file) for file in os.listdir(dir) if file.split('.')[-1].lower() in IMAGE_EXTs or os.path.isdir(os.path.join(dir, file))])

if __name__ == '__main__':
    images = list_images(FOLDER_IN_PATH)

    for i, path in enumerate(images):
        print(f'[{i}/{len(images)}]Processing {path}...')
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        mask = 1.0 - np.tile(((img > 100).astype(np.int32).sum(axis=-1) == 3)[..., None], (1, 1, 3)).astype(np.float32)
        img = (img * mask).astype(np.uint8)

        
        cv2.imwrite(os.path.join(FOLDER_OUT_PATH, os.path.basename(path)), img)

    print('!! DONE !!')
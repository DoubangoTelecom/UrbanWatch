import os, torch
from PIL import Image, ImageOps, ImageChops
from rich import print


FOLDER_PATH_IN = 'E:/Projects/ocr_datasets/alpr/garbage'
FOLDER_PATH_OUT = 'E:/Projects/ocr_datasets/alpr/garbage2'
IMAGE_EXTs = ['jpg', 'jpeg']

assert(FOLDER_PATH_IN != FOLDER_PATH_OUT)

files = [os.path.join(FOLDER_PATH_IN, file) for file in os.listdir(FOLDER_PATH_IN) if os.path.isfile(os.path.join(FOLDER_PATH_IN, file))]

back_img = Image.new('RGB', (150, 150))
for index, file in enumerate(files):
    print(f'Processing {index}/{len(files)}...')
    img = Image.open(file).convert('RGB')
    if img.size[0] == 300 and img.size[1] == 150:            
        img = img.crop((0, 0, 150, 150))
        diff = ImageChops.difference(img, back_img)
        diff = ImageChops.add(diff, diff, 2.0, -20)
        bbox = diff.getbbox()
        if bbox:
            img = img.crop(bbox)
        
    img = ImageOps.contain(img, (150, 150)) # 150 is from old code and max of what would be used in the future    

    img.save(os.path.join(FOLDER_PATH_OUT, os.path.basename(file)))

print('!!! DONE !!!')
    
    
    
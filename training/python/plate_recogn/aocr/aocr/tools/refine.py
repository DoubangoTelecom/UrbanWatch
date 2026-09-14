import os
from PIL import Image, ImageOps, ImageChops
from aocr.config import Config
import numpy as np
import onnxruntime as rt

MODEL_PATH = 'C:/Projects/Github/ultimate/UrbanWatch/training/python/plate_recogn/aocr/inference_models/aocr_anpr_latin.onnx'
CONFIG_PATH = 'C:/Projects/GitHub/ultimate/UrbanWatch/training/python/plate_recogn/aocr/configs/config_latin.yml'
IMAGE_EXTs = ['tif','png','jpg','jpeg','bmp']

FOLDER_PATH_IN = 'E:/Projects/ocr_datasets/alpr/garbage/'
FOLDER_PATH_OUT = 'E:/Projects/ocr_datasets/alpr/garbage2'
IMAGE_EXTs = ['jpg', 'jpeg']
XML_TEMPLATE = '''<annotation>	<filename>{}</filename>	<size>		<width>{}</width>		<height>{}</height>		<depth>3</depth>	</size>	<object>	
<text>{}</text></object></annotation>'''

assert(FOLDER_PATH_IN != FOLDER_PATH_OUT)

cfg = Config.parse(CONFIG_PATH)
files = [os.path.join(FOLDER_PATH_IN, name) for name in os.listdir(FOLDER_PATH_IN) if name.split('.')[-1].lower() in IMAGE_EXTs]
assert len(files) > 0, f'Folder ({FOLDER_PATH_IN}) is empty'
back_img = Image.new('RGB', (150, 150))

model = rt.InferenceSession(MODEL_PATH, providers=rt.get_available_providers())
input = model.get_inputs()[0].name

# alphabet
alphabet_list = ['$'] + list(cfg.model.alphabet) # $ at 0 is CTC blank

def pred_decode(alphabet :list, codes :np.array, ctc :bool):
    assert(isinstance(alphabet, list))
    assert(len(codes.shape) == 1)
    char_list = []
    for i in range(len(codes)):
        if codes[i] != 0 and (not (i > 0 and codes[i - 1] == codes[i]) or not ctc):
            char_list.append(alphabet[codes[i]])

    return ''.join(char_list)

def image_preprocess(image: Image, cfg, channel_first :bool=True):
    target_size = (cfg.model.imgW, cfg.model.imgH)
    
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

for index, img_path in enumerate(files):
    print(f'Processing {index}/{len(files)}...')

    img = Image.open(img_path).convert('RGB')
    assert img.size == (300, 150), 'Invalid size'
    img = img.crop((0, 0, 150, 150))
    diff = ImageChops.difference(img, back_img)
    diff = ImageChops.add(diff, diff, 2.0, -20)
    bbox = diff.getbbox()
    if bbox:
        img = img.crop(bbox)

    # OCR
    preds = model.run(None, {input: image_preprocess(img, cfg, channel_first=True)})[0]
    preds_index = np.argmax(preds, axis=-1)
    preds_max_prob = np.take_along_axis(preds, preds_index[...,None], axis=2)
    preds_str = pred_decode(alphabet_list, preds_index.flatten(), ctc=cfg.train.loss.type == 'ctc')

    # Write result    
    bparts = os.path.basename(img_path).split('__')
    assert(len(bparts) > 1)
    bname = '__'.join(bparts[1:])
    fname = f"{preds_str}__{'.'.join(bname.split('.')[:-1])}"            
    img_name = f'{fname}.jpg'
    img.save(os.path.join(FOLDER_PATH_OUT, img_name))
    xml_data = XML_TEMPLATE.format(img_name, img.size[0], img.size[1], preds_str)
    with open(os.path.join(FOLDER_PATH_OUT, f'{fname}.xml'), "w") as xml_file:
        xml_file.write(xml_data)
            

print('!!! DONE !!!')
    
    
    
import os, torch
from PIL import Image, ImageOps, ImageChops
from rich import print
from aocr.config import Config
from aocr.utils import CTCLabelConverter, CELabelConverter
from aocr.model import AOCR
from aocr.dataset import RawDataset, AlignCollate

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

MODEL_PATH = 'C:/Projects/GitHub/ultimate/UrbanWatch/training/python/plate_recogn/aocr/saved_models/anpr_latin/best_norm_ED.pth'
CONFIG_PATH = 'C:/Projects/GitHub/ultimate/UrbanWatch/training/python/plate_recogn/aocr/configs/config_latin.yml'

FOLDER_PATH_IN = 'E:/Projects/ocr_datasets/alpr/garbage2'
FOLDER_PATH_OUT = 'E:/Projects/ocr_datasets/alpr/garbage'
IMAGE_EXTs = ['jpg', 'jpeg']
XML_TEMPLATE = '''<annotation>	<filename>{}</filename>	<size>		<width>{}</width>		<height>{}</height>		<depth>3</depth>	</size>	<object>	
<text>{}</text></object></annotation>'''

assert(FOLDER_PATH_IN != FOLDER_PATH_OUT)

# Prepare Engine
cfg = Config.parse(CONFIG_PATH)
converter = CTCLabelConverter(cfg.model.alphabet) if cfg.train.loss.type == 'ctc' else \
        CELabelConverter(cfg.model.alphabet)
model = AOCR(cfg, training=False).to(device).eval()
model.load_state_dict(torch.load(MODEL_PATH, map_location=device), strict=True)

# prepare data. two demo images from https://github.com/bgshih/crnn#run-demo
AlignCollate_demo = AlignCollate(cfg)
demo_data = RawDataset(root=FOLDER_PATH_IN, opt=cfg)  # use RawDataset
demo_loader = torch.utils.data.DataLoader(
    demo_data, batch_size=1,
    shuffle=False,
    num_workers=0,
    collate_fn=AlignCollate_demo, pin_memory=True)

index = 0
for image_tensors, image_path_list in demo_loader:
    print(f'Processing {index}/{len(demo_loader)}...')
    batch_size = image_tensors.size(0)
    image = image_tensors.to(device)
    
    # Inference
    preds = model(image)

    # Select max probabilty (greedy decoding) then decode index to character
    preds_size = torch.IntTensor([preds.size(1)] * batch_size)
    preds_max_prob, preds_index = preds.max(-1)
    preds_str = converter.decode(preds_index, preds_size)

    back_img = Image.new('RGB', (150, 150))
    for img_path, pred, pred_max_prob in zip(image_path_list, preds_str, preds_max_prob):
        img = Image.open(img_path).convert('RGB')
        img = ImageOps.contain(img, (150, 150)) # 150 is from old code and max of what would be used in the future
        
        bparts = os.path.basename(img_path).split('__')
        assert(len(bparts) > 1)
        bname = '__'.join(bparts[1:])
        fname = f"{pred}__{'.'.join(bname.split('.')[:-1])}"            
        img_name = f'{fname}.jpg'
        img.save(os.path.join(FOLDER_PATH_OUT, img_name))
        xml_data = XML_TEMPLATE.format(img_name, img.size[0], img.size[1], pred)
        with open(os.path.join(FOLDER_PATH_OUT, f'{fname}.xml'), "w") as xml_file:
            xml_file.write(xml_data)
            
    index += 1

print('!!! DONE !!!')
    
    
    
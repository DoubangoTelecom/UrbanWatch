import os, coloredlogs, logging, random, numpy as np, io, shutil, json
from pymage_size import get_image_size
from bs4 import BeautifulSoup

VAL_RATIO = 0.1
SOURCE_FOLDER = '/home/garbage/mnt/Projects/tensorflow-models/research/object_detection/images'
DEST_FOLDER = '/home/projects/urban-watch/plate_detect/nanodet/datasets/anpr_pysearch'

IMAGE_EXTs = ['tif','png','jpg','jpeg','bmp']

CLASSES = {
    'plate': 1, 
}

FOLDERS = [
    "test/",
    "train/",
]

coco_annotations_list = { 'train': [], 'val': [] }
coco_images_list = { 'train': [], 'val': [] }

if __name__ == '__main__':
    coloredlogs.install(level='INFO')
    logger = logging.getLogger('org.doubango.urbanwatch.dataset_voc2coco')
    logger.info('Building dataset...')
    val_pop = list(np.arange(0, 1, VAL_RATIO))
    clip_fn = lambda minn, val, maxx: min(max(minn, val), maxx)
    
    # Make sure output directory is empty
    coco_base_folder = os.path.join(DEST_FOLDER, 'coco')
    assert(os.path.isdir(DEST_FOLDER))
    if os.path.isdir(coco_base_folder) and len(os.listdir(coco_base_folder)) > 0:
        logger.error('Output directory ({}) must be empty'.format(coco_base_folder))
        raise Exception('{} not empty'.format(coco_base_folder))
    
    # Create folders
    coco_folders = {
        'train': os.path.join(coco_base_folder, 'train2017'),
        'val': os.path.join(coco_base_folder, 'val2017')
    }
    os.makedirs(os.path.join(coco_base_folder, 'annotations'))
    os.makedirs(coco_folders['train'])
    os.makedirs(coco_folders['val'])
    
    for i, folder in enumerate(FOLDERS):
        logger.info('Processing [%d/%d]...', i, len(FOLDERS))
        folder_fullpath = os.path.join(SOURCE_FOLDER, folder)
        files = [os.path.join(folder_fullpath, file) for file in os.listdir(folder_fullpath) if os.path.isfile(os.path.join(folder_fullpath, file))]
        random.shuffle(files)
        for j, file in enumerate(files):                
            fparts = file.split('.')
            ext = fparts[-1].lower()
            if not ext in IMAGE_EXTs:
                continue
            
            xml_file = '.'.join(fparts[:-1]) + '.xml'
            if not os.path.isfile(xml_file):
                logger.error('{} do not exist'.format(xml_file))
                continue
            
            dom = BeautifulSoup(io.open(xml_file, mode="r", encoding="utf-8").read(), features='xml')
            annotations = dom.findAll('annotation')
            if annotations is None:
                logger.warn('No annotations in {}'.format(file))
                continue
            
            # Get image size
            img_format = get_image_size(file)
            image_width, image_height = img_format.get_dimensions()
            
            target = 'val' if (random.choice(val_pop) == 0) else 'train'
            
            # Copy the image
            image_id = len(coco_images_list[target]) + 1
            image_name = '{}.{}'.format(image_id, ext)
            shutil.copy(file, os.path.join(coco_folders[target], image_name))
            coco_images_list[target].append(
                {
                    "id": image_id,
                    "file_name": image_name,
                    "height": image_height,
                    "width": image_width
                }
            )
            
            # Annotations
            for annotation in annotations:
                # get size
                size = annotation.find('size')
                width = int(size.find('width').text)
                height = int(size.find('height').text)
                if width != image_width or height != image_height:
                    logger.warn('Image size mismatch: {}<>{} or {}<>{}', width, image_width, height, image_height)
                
                objects = annotation.findAll('object')
                if objects is not None:
                    for object in objects:
                        name = object.find('name')
                        if name.text in CLASSES:
                            bndbox = object.find('bndbox')
                            x1, y1, x2, y2 = clip_fn(0, int(bndbox.find('xmin').text), width-1), \
                                clip_fn(0, int(bndbox.find('ymin').text), height-1), \
                                clip_fn(0, int(bndbox.find('xmax').text), width-1), \
                                clip_fn(0, int(bndbox.find('ymax').text), height-1)
                            if x1 < x2 and y1 < y2:
                                w, h = (x2 - x1), (y2 - y1)             
                                coco_annotations_list[target].append(
                                    {
                                        "id": len(coco_annotations_list[target]) + 1,
                                        "image_id": image_id,
                                        "category_id": CLASSES[name.text],
                                        "iscrowd": 0,
                                        "area": w*h,
                                        "bbox": [ x1, y1, w, h ] # x1, x2, width, height
                                    }
                                )
                                
    
    # Write annotations
    for target in coco_annotations_list.keys():
        coco_data = {
            "info": target,
            "licenses": "AGPL",
            "annotations": coco_annotations_list[target],
            "images": coco_images_list[target],
            "categories": [{"id": class_id, "name": class_name} for class_name, class_id in CLASSES.items()]
        }
        with open(os.path.join(coco_base_folder, 'annotations', 'instances_{}.json'.format(target)), "w") as f:
            json.dump(coco_data, f)
            
    logger.info('!!DONE!!')
    
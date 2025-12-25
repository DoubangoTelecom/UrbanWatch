import os, coloredlogs, logging, random, numpy as np, io, shutil, json
from pymage_size import get_image_size
from bs4 import BeautifulSoup

VAL_RATIO = 0.1
SOURCE_FOLDER = '/home/urban-watch/plate_detect/dataset_voc'#'E:/Projects/ocr_datasets/alpr/detect2'
DEST_FOLDER = '/home/urban-watch/plate_detect/YOLOX/datasets' #'E:/Projects/Garbage/plate_detect_dataset'

IMAGE_EXTs = ['tif','png','jpg','jpeg','bmp']

CLASSES = {
    'plate': 1, 
    'car': 2
}

# Comes from https://github.com/DoubangoTelecom/ultimateALPR/blob/6832b3b1e5e35942723c1b32a6aef751ddf861e2/training/detect2_copy_files.cxx#L38
FOLDERS = [
    ## Google images ##
    "motorcycle/",
    "google images/",
    "notacar/",

    ## Just to have fun ##
    "anpr-dataset-tunisian-plates-and-digits/",

    ## Extrats ##
    "extras/",

    ## Camera ##
    "camera",

    ## Cities ##
    "cities/london/",

    ## Coco (first #500), Negatives ##
    "coco/",

    ## Changed from png to jpg ##
    "png_to_jpg/",

    ## Strike (Google images), Negatives ##
    "strike/",

    ## Croatia ##
    "croatia/",

    ## Greece ##
    "greece/difficult_dirt_shadows/",
    "greece/difficult_shadows/",
    "greece/difficult_tracks_night/",

    ## India ##
    "india/",

    ## Kaggle ##
    "kaggle/training_set/",

    ## Old ##
    "old/",

    ## OpenALPR ##
    "openalpr/br/",
    "openalpr/eu/",
    "openalpr/us/",

    ## From customers ##
    "indonesia/",
    "siri lanka/",
    "omnivisionseguridad/",
    "ismail/nigeria/",
    "nick_bolton/new_zeland",
    "nikita/kzh/pic",
    "john/korea/",
    "acerits/taiwan/",
    "avadesign/taiwan/",
    "skycopinc-tennessee/0/",
    "skycopinc-tennessee/1/",
    "skycopinc-tennessee/2/",
    "skycopinc-tennessee/3/",
    "skycopinc-tennessee/4/",
    "seguritech/imagenes_carros/0",
    "seguritech/imagenes_carros/1",
    "seguritech/imagenes_carros/2",
    "seguritech/imagenes_carros/3",
    "seguritech/imagenes_carros/4",
    "seguritech/imagenes_carros/5",
    "seguritech/imagenes_carros_edo/0",
    "seguritech/imagenes_carros_edo/1",
    "seguritech/imagenes_carros_edo/2",
# // Not done yet
#	"seguritech/imagenes_carros_edo/3", // TODO(dmi)
#	"seguritech/imagenes_carros_edo/4", // TODO(dmi)
#	"seguritech/imagenes_carros_edo/5", // TODO(dmi)
#	"seguritech/imagenes_carros_edo/6", // TODO(dmi)
#	

    ## vizura.net ##
# // Do not include, partial cars
#	"vizura.net/TrainingSet1/000/",
#	"vizura.net/TrainingSet1/001/",
#	"vizura.net/TrainingSet1/002/",
#	"vizura.net/TrainingSet1/003/",
#	"vizura.net/TrainingSet1/004/",
#	"vizura.net/TrainingSet1/005/",
#	"vizura.net/TrainingSet1/006/",
#	"vizura.net/TrainingSet1/007/",
#	"vizura.net/TrainingSet1/008/",
#	"vizura.net/TrainingSet1/009/",
#
    
    ## platesmania ##
    "platesmania/ae/",
    "platesmania/am/",
    "platesmania/at/",
    "platesmania/az/",
    "platesmania/be/",
    "platesmania/bg/",
    "platesmania/ca/",
    "platesmania/ch/",
    "platesmania/cn/",
    "platesmania/cz/",
    "platesmania/de/",
    "platesmania/dk/",
    "platesmania/fi/",
    "platesmania/fr/",
    "platesmania/gr/",
    "platesmania/hr/",
    "platesmania/il/",
    "platesmania/it/",
    "platesmania/kz/",
    "platesmania/lu/",
    "platesmania/mc/",
    "platesmania/me/",
    "platesmania/no/",
    "platesmania/pl/",
    "platesmania/pt/",
    "platesmania/ro/",
    "platesmania/rs/",
    "platesmania/ru/",
    "platesmania/se/",
    "platesmania/si/",
    "platesmania/sk/",
    "platesmania/su/",
    "platesmania/ua/",
    "platesmania/uk/",
    "platesmania/us/",
    "platesmania/uz/",
    "platesmania/xx/",
    
    ## olavsplates (africa) ##
    "olavsplates/africa/algeria/",
    "olavsplates/africa/angola/",
    "olavsplates/africa/benin/",
    "olavsplates/africa/gambia/",
    "olavsplates/africa/ghana/",
    "olavsplates/africa/morocco/",
    "olavsplates/africa/seychelles/",
    "olavsplates/africa/sierra leone/",
    "olavsplates/africa/south africa/",
    "olavsplates/africa/tristan da cunha/",
    "olavsplates/africa/tunisia/",
    "olavsplates/africa/uganda/",
    "olavsplates/africa/zambia/",

    ## olavsplates (americas) ##
    "olavsplates/americas/argentina/",
    "olavsplates/americas/brazil/",
    "olavsplates/americas/canada/",
    "olavsplates/americas/chile/",
    "olavsplates/americas/guatemala/",
    "olavsplates/americas/martinique/",
    "olavsplates/americas/mexico/",
    "olavsplates/americas/netherlands antilles/",
    "olavsplates/americas/paraguay/",
    "olavsplates/americas/usa/",

    ## olavsplates (asia) ##
    "olavsplates/asia/armenia/",
    "olavsplates/asia/azerbaijan/",
    "olavsplates/asia/cambodia/",
    "olavsplates/asia/china/",
    "olavsplates/asia/cyprus/",
    "olavsplates/asia/georgia/",
    "olavsplates/asia/hong kong/",
    "olavsplates/asia/india/",
    "olavsplates/asia/indonesia/",
    "olavsplates/asia/iran/",
    "olavsplates/asia/iraq/",
    "olavsplates/asia/israel/",
    "olavsplates/asia/japan/",
    "olavsplates/asia/kazakhstan/",
    "olavsplates/asia/korea/",
    "olavsplates/asia/kuwait/",
    "olavsplates/asia/lebanon/",
    "olavsplates/asia/malaysia/",
    "olavsplates/asia/nepal/",
    "olavsplates/asia/oman/",
    "olavsplates/asia/pakistan/",
    "olavsplates/asia/philipines/",
    "olavsplates/asia/qatar/",
    "olavsplates/asia/saudi arabia/",
    "olavsplates/asia/singapore/",
    "olavsplates/asia/united arab emirates/",

    ## olavsplates (europe) ##
    "olavsplates/europe/albania/",
    "olavsplates/europe/alderney/",
    "olavsplates/europe/andorra/",
    "olavsplates/europe/austria/",
    "olavsplates/europe/belarus/",
    "olavsplates/europe/belgium/",
    "olavsplates/europe/bosnia and herzegovina/",
    "olavsplates/europe/bulgaria/",
    "olavsplates/europe/croatia/",
    "olavsplates/europe/czetia/",
    "olavsplates/europe/denmark/",
    "olavsplates/europe/estonia/",
    "olavsplates/europe/faroe islands/",
    "olavsplates/europe/finland/",
    "olavsplates/europe/france/",
    "olavsplates/europe/germany/",
    "olavsplates/europe/gibraltar/",
    "olavsplates/europe/greece/",
    "olavsplates/europe/guernsey/",
    "olavsplates/europe/hungary/",
    "olavsplates/europe/iceland/",
    "olavsplates/europe/ireland/",
    "olavsplates/europe/isle of man/",
    "olavsplates/europe/italy/",
    "olavsplates/europe/jersey/",
    "olavsplates/europe/latvia/",
    "olavsplates/europe/liechtenstein/",
    "olavsplates/europe/lithuania/",
    "olavsplates/europe/luxembourg/",
    "olavsplates/europe/macedonia/",
    "olavsplates/europe/malta/",
    "olavsplates/europe/moldova/",
    "olavsplates/europe/monaco/",
    "olavsplates/europe/montenegro/",
    "olavsplates/europe/netherlands/",
    "olavsplates/europe/norway/",
    "olavsplates/europe/poland/",
    "olavsplates/europe/portugal/",
    "olavsplates/europe/romania/",
    "olavsplates/europe/russia/",
    "olavsplates/europe/sanmarino/",
    "olavsplates/europe/serbia/",
    "olavsplates/europe/slovakia/",
    "olavsplates/europe/slovenia/",
    "olavsplates/europe/spain/",
    "olavsplates/europe/sweden/",
    "olavsplates/europe/switzerland/",
    "olavsplates/europe/turkey/",
    "olavsplates/europe/ukraine/",
    "olavsplates/europe/united kingdom/",
    "olavsplates/europe/vatican city state/",

    ## olavsplates (oceania) ##
    "olavsplates/oceania/australia/",
    "olavsplates/oceania/new zeland/",
    "olavsplates/oceania/vanuatu/",
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
    
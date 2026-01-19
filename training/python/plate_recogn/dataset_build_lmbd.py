import os, coloredlogs, logging, random, numpy as np, io, lmdb, shutil
from bs4 import BeautifulSoup
from PIL import Image, ImageChops

VAL_RATIO = 0.1
DEST_FOLDER = '/home/urban-watch/plate_recogn/aocr/datasets/latin'
CACHE_SIZE = 1e9

IMAGE_EXTs = ['tif','png','jpg','jpeg','bmp'] 

# Comes from https://github.com/DoubangoTelecom/ultimateALPR/blob/6832b3b1e5e35942723c1b32a6aef751ddf861e2/training/recogn_copy_files.cxx#L61
FOLDERS = [
    ## Collected ##
	"/media/mamadou/TOSHIBA EXT/Projects/ocr_datasets/alpr/recogn/latin/collected/000/",
	"/media/mamadou/TOSHIBA EXT/Projects/ocr_datasets/alpr/recogn/latin/collected/001/",

	## From customers ##
	"/media/mamadou/TOSHIBA EXT/Projects/ocr_datasets/alpr/recogn/latin/mexico/seguritech/imagenes_carros/",
	"/media/mamadou/TOSHIBA EXT/Projects/ocr_datasets/alpr/recogn/latin/siri_lanka/elogiclanka/",
	"/media/mamadou/TOSHIBA EXT/Projects/ocr_datasets/alpr/recogn/latin/turkey/noveltybilisim/",
	"/media/mamadou/TOSHIBA EXT/Projects/ocr_datasets/alpr/recogn/latin/new_zeland/nick_bolton/",
	"/media/mamadou/TOSHIBA EXT/Projects/ocr_datasets/alpr/recogn/latin/taiwan/acerits/",
	"/media/mamadou/TOSHIBA EXT/Projects/ocr_datasets/alpr/recogn/latin/taiwan/avadesign/",
	"/media/mamadou/TOSHIBA EXT/Projects/ocr_datasets/alpr/recogn/latin/usa/skycopinc-tennessee/",
	"/media/mamadou/TOSHIBA EXT/Projects/ocr_datasets/alpr/recogn/latin/usa/stacked/",
	"/media/mamadou/TOSHIBA EXT/Projects/ocr_datasets/alpr/recogn/latin/indonesia/",
	"/media/mamadou/TOSHIBA EXT/Projects/ocr_datasets/alpr/recogn/latin/myriade/",

	## Scrapping ##
	"/media/mamadou/TOSHIBA EXT/Projects/ocr_datasets/alpr/recogn/latin/scrap/platesmania/brazil/",
	"/media/mamadou/TOSHIBA EXT/Projects/ocr_datasets/alpr/recogn/latin/scrap/platesmania/canada/",
	"/media/mamadou/TOSHIBA EXT/Projects/ocr_datasets/alpr/recogn/latin/scrap/platesmania/usa/",
	"/media/mamadou/TOSHIBA EXT/Projects/ocr_datasets/alpr/recogn/latin/scrap/platesmania/kazakhstan/",
	"/media/mamadou/TOSHIBA EXT/Projects/ocr_datasets/alpr/recogn/latin/scrap/platesmania/turkey/",

	## Google images ##
	"/media/mamadou/TOSHIBA EXT/Projects/ocr_datasets/alpr/recogn/latin/google images/",
	"/media/mamadou/TOSHIBA EXT/Projects/ocr_datasets/alpr/recogn/latin/siri_lanka/googleimage/",

	## Extrats ##
	"/media/mamadou/TOSHIBA EXT/Projects/ocr_datasets/alpr/recogn/latin/extras/",

	## Inverse (Ooops) ##
	"/media/mamadou/TOSHIBA EXT/Projects/ocr_datasets/alpr/recogn/latin/inverse/",
	
	## Cities ##
	"/media/mamadou/TOSHIBA EXT/Projects/ocr_datasets/alpr/recogn/latin/cities/london/",
	
	## Changed from png to jpg ##
	"/media/mamadou/TOSHIBA EXT/Projects/ocr_datasets/alpr/recogn/latin/png_to_jpg/",

	## Croatia ##
	"/media/mamadou/TOSHIBA EXT/Projects/ocr_datasets/alpr/recogn/latin/croatia/",

	## Greece ##
	"/media/mamadou/TOSHIBA EXT/Projects/ocr_datasets/alpr/recogn/latin/greece/difficult_dirt_shadows/",
	"/media/mamadou/TOSHIBA EXT/Projects/ocr_datasets/alpr/recogn/latin/greece/difficult_shadows/",
	"/media/mamadou/TOSHIBA EXT/Projects/ocr_datasets/alpr/recogn/latin/greece/difficult_tracks_night/",
 
	## Brazil ##
	"/media/mamadou/TOSHIBA EXT/Projects/ocr_datasets/alpr/recogn/latin/brazil/day/",
	"/media/mamadou/TOSHIBA EXT/Projects/ocr_datasets/alpr/recogn/latin/brazil/night/",

	## India ##
	"/media/mamadou/TOSHIBA EXT/Projects/ocr_datasets/alpr/recogn/latin/india/",

	## Tunisia ##
	"/media/mamadou/TOSHIBA EXT/Projects/ocr_datasets/alpr/recogn/latin/tunisia/",
	
	## Kaggle ##
	"/media/mamadou/TOSHIBA EXT/Projects/ocr_datasets/alpr/recogn/latin/kaggle/training_set/",

	## Old ##
	"/media/mamadou/TOSHIBA EXT/Projects/ocr_datasets/alpr/recogn/latin/old/",

	## OpenALPR ##
	"/media/mamadou/TOSHIBA EXT/Projects/ocr_datasets/alpr/recogn/latin/openalpr/br/",
	"/media/mamadou/TOSHIBA EXT/Projects/ocr_datasets/alpr/recogn/latin/openalpr/eu/",
	"/media/mamadou/TOSHIBA EXT/Projects/ocr_datasets/alpr/recogn/latin/openalpr/us/",

	## Google Search ##
	"/media/mamadou/TOSHIBA EXT/Projects/ocr_datasets/alpr/recogn/latin/websites/www.google.com - search/",
	"/media/mamadou/TOSHIBA EXT/Projects/ocr_datasets/alpr/recogn/latin/websites/www.google.com - search - us - mosaic/",

	## alpr-lpci-google-images-download ##
	"/media/mamadou/TOSHIBA EXT/Projects/ocr_datasets/alpr/recogn/latin/alpr-lpci-google-images-download/United_Arab_Emirates-Abu_Dhabi",
	"/media/mamadou/TOSHIBA EXT/Projects/ocr_datasets/alpr/recogn/latin/alpr-lpci-google-images-download/United_Arab_Emirates-Dubai",
	"/media/mamadou/TOSHIBA EXT/Projects/ocr_datasets/alpr/recogn/latin/alpr-lpci-google-images-download/Kyrgyzstan",
	"/media/mamadou/TOSHIBA EXT/Projects/ocr_datasets/alpr/recogn/latin/alpr-lpci-google-images-download/Kazakhstan",
	"/media/mamadou/TOSHIBA EXT/Projects/ocr_datasets/alpr/recogn/latin/alpr-lpci-google-images-download/Uzbekistan",
	"/media/mamadou/TOSHIBA EXT/Projects/ocr_datasets/alpr/recogn/latin/alpr-lpci-google-images-download/Turkmenistan",
	"/media/mamadou/TOSHIBA EXT/Projects/ocr_datasets/alpr/recogn/latin/alpr-lpci-google-images-download/Tajikistan",
	"/media/mamadou/TOSHIBA EXT/Projects/ocr_datasets/alpr/recogn/latin/alpr-lpci-google-images-download/Spain",
	"/media/mamadou/TOSHIBA EXT/Projects/ocr_datasets/alpr/recogn/latin/alpr-lpci-google-images-download/USA-Florida",
	"/media/mamadou/TOSHIBA EXT/Projects/ocr_datasets/alpr/recogn/latin/alpr-lpci-google-images-download/USA-Utah",
	"/media/mamadou/TOSHIBA EXT/Projects/ocr_datasets/alpr/recogn/latin/alpr-lpci-google-images-download/USA-California",
	"/media/mamadou/TOSHIBA EXT/Projects/ocr_datasets/alpr/recogn/latin/alpr-lpci-google-images-download/Russia",
	"/media/mamadou/TOSHIBA EXT/Projects/ocr_datasets/alpr/recogn/latin/alpr-lpci-google-images-download/Brasil",

#	if 0 // Not done yet
	#"/media/mamadou/TOSHIBA EXT/Projects/ocr_datasets/alpr/recogn/latin/alpr-lpci-google-images-download/United_Arab_Emirates-Dubai",
	#"/media/mamadou/TOSHIBA EXT/Projects/ocr_datasets/alpr/recogn/latin/alpr-lpci-google-images-download/United_Arab_Emirates-Ajman",
	#"/media/mamadou/TOSHIBA EXT/Projects/ocr_datasets/alpr/recogn/latin/alpr-lpci-google-images-download/United_Arab_Emirates-Fujairah",
	#"/media/mamadou/TOSHIBA EXT/Projects/ocr_datasets/alpr/recogn/latin/alpr-lpci-google-images-download/United_Arab_Emirates-Ras_al_Khaimah",
	#"/media/mamadou/TOSHIBA EXT/Projects/ocr_datasets/alpr/recogn/latin/alpr-lpci-google-images-download/United_Arab_Emirates-Sharjah",
	#"/media/mamadou/TOSHIBA EXT/Projects/ocr_datasets/alpr/recogn/latin/alpr-lpci-google-images-download/United_Arab_Emirates-Umm_al_Qaiwain",
#	endif

	## olavsplates ##
	"/media/mamadou/TOSHIBA EXT/Projects/ocr_datasets/alpr/recogn/latin/websites/www.olavsplates.com/000/",
	"/media/mamadou/TOSHIBA EXT/Projects/ocr_datasets/alpr/recogn/latin/websites/www.olavsplates.com/001/",
	"/media/mamadou/TOSHIBA EXT/Projects/ocr_datasets/alpr/recogn/latin/websites/www.olavsplates.com/002/",
	"/media/mamadou/TOSHIBA EXT/Projects/ocr_datasets/alpr/recogn/latin/websites/www.olavsplates.com/003/",
	"/media/mamadou/TOSHIBA EXT/Projects/ocr_datasets/alpr/recogn/latin/websites/www.olavsplates.com/004/",
	"/media/mamadou/TOSHIBA EXT/Projects/ocr_datasets/alpr/recogn/latin/websites/www.olavsplates.com/005/",
	"/media/mamadou/TOSHIBA EXT/Projects/ocr_datasets/alpr/recogn/latin/websites/www.olavsplates.com/006/",
	"/media/mamadou/TOSHIBA EXT/Projects/ocr_datasets/alpr/recogn/latin/websites/www.olavsplates.com/007/",

	## vizura.net ##
	"/media/mamadou/TOSHIBA EXT/Projects/ocr_datasets/alpr/recogn/latin/vizura.net/TrainingSet1/000/",
	"/media/mamadou/TOSHIBA EXT/Projects/ocr_datasets/alpr/recogn/latin/vizura.net/TrainingSet1/001/",
	"/media/mamadou/TOSHIBA EXT/Projects/ocr_datasets/alpr/recogn/latin/vizura.net/TrainingSet1/002/",
	"/media/mamadou/TOSHIBA EXT/Projects/ocr_datasets/alpr/recogn/latin/vizura.net/TrainingSet1/003/",
	"/media/mamadou/TOSHIBA EXT/Projects/ocr_datasets/alpr/recogn/latin/vizura.net/TrainingSet1/004/",
	"/media/mamadou/TOSHIBA EXT/Projects/ocr_datasets/alpr/recogn/latin/vizura.net/TrainingSet1/005/",
	"/media/mamadou/TOSHIBA EXT/Projects/ocr_datasets/alpr/recogn/latin/vizura.net/TrainingSet1/006/",
	"/media/mamadou/TOSHIBA EXT/Projects/ocr_datasets/alpr/recogn/latin/vizura.net/TrainingSet1/007/",
	"/media/mamadou/TOSHIBA EXT/Projects/ocr_datasets/alpr/recogn/latin/vizura.net/TrainingSet1/008/",
	"/media/mamadou/TOSHIBA EXT/Projects/ocr_datasets/alpr/recogn/latin/vizura.net/TrainingSet1/009/",

	## platesmania ##
	"/media/mamadou/TOSHIBA EXT/Projects/ocr_datasets/alpr/recogn/latin/websites/www.platesmania.com/ae/",
	"/media/mamadou/TOSHIBA EXT/Projects/ocr_datasets/alpr/recogn/latin/websites/www.platesmania.com/at/",
	"/media/mamadou/TOSHIBA EXT/Projects/ocr_datasets/alpr/recogn/latin/websites/www.platesmania.com/az/",
	"/media/mamadou/TOSHIBA EXT/Projects/ocr_datasets/alpr/recogn/latin/websites/www.platesmania.com/be/",
	"/media/mamadou/TOSHIBA EXT/Projects/ocr_datasets/alpr/recogn/latin/websites/www.platesmania.com/bg/",
	"/media/mamadou/TOSHIBA EXT/Projects/ocr_datasets/alpr/recogn/latin/websites/www.platesmania.com/ca/",
	"/media/mamadou/TOSHIBA EXT/Projects/ocr_datasets/alpr/recogn/latin/websites/www.platesmania.com/ch/",
	"/media/mamadou/TOSHIBA EXT/Projects/ocr_datasets/alpr/recogn/latin/websites/www.platesmania.com/de/",
	"/media/mamadou/TOSHIBA EXT/Projects/ocr_datasets/alpr/recogn/latin/websites/www.platesmania.com/dk/",
	"/media/mamadou/TOSHIBA EXT/Projects/ocr_datasets/alpr/recogn/latin/websites/www.platesmania.com/fi/",
	"/media/mamadou/TOSHIBA EXT/Projects/ocr_datasets/alpr/recogn/latin/websites/www.platesmania.com/fr/",
	"/media/mamadou/TOSHIBA EXT/Projects/ocr_datasets/alpr/recogn/latin/websites/www.platesmania.com/gr/",
	"/media/mamadou/TOSHIBA EXT/Projects/ocr_datasets/alpr/recogn/latin/websites/www.platesmania.com/hr/",
	"/media/mamadou/TOSHIBA EXT/Projects/ocr_datasets/alpr/recogn/latin/websites/www.platesmania.com/il/",
	"/media/mamadou/TOSHIBA EXT/Projects/ocr_datasets/alpr/recogn/latin/websites/www.platesmania.com/it/",
	"/media/mamadou/TOSHIBA EXT/Projects/ocr_datasets/alpr/recogn/latin/websites/www.platesmania.com/kz/",
	"/media/mamadou/TOSHIBA EXT/Projects/ocr_datasets/alpr/recogn/latin/websites/www.platesmania.com/lu/",
	"/media/mamadou/TOSHIBA EXT/Projects/ocr_datasets/alpr/recogn/latin/websites/www.platesmania.com/mc/",
	"/media/mamadou/TOSHIBA EXT/Projects/ocr_datasets/alpr/recogn/latin/websites/www.platesmania.com/me/",
	"/media/mamadou/TOSHIBA EXT/Projects/ocr_datasets/alpr/recogn/latin/websites/www.platesmania.com/ml/",
	"/media/mamadou/TOSHIBA EXT/Projects/ocr_datasets/alpr/recogn/latin/websites/www.platesmania.com/mx/",
	"/media/mamadou/TOSHIBA EXT/Projects/ocr_datasets/alpr/recogn/latin/websites/www.platesmania.com/no/",
	"/media/mamadou/TOSHIBA EXT/Projects/ocr_datasets/alpr/recogn/latin/websites/www.platesmania.com/pl/",
	"/media/mamadou/TOSHIBA EXT/Projects/ocr_datasets/alpr/recogn/latin/websites/www.platesmania.com/pt/",
	"/media/mamadou/TOSHIBA EXT/Projects/ocr_datasets/alpr/recogn/latin/websites/www.platesmania.com/ro/",
	"/media/mamadou/TOSHIBA EXT/Projects/ocr_datasets/alpr/recogn/latin/websites/www.platesmania.com/rs/",
	"/media/mamadou/TOSHIBA EXT/Projects/ocr_datasets/alpr/recogn/latin/websites/www.platesmania.com/ru/",
	"/media/mamadou/TOSHIBA EXT/Projects/ocr_datasets/alpr/recogn/latin/websites/www.platesmania.com/se/",
	"/media/mamadou/TOSHIBA EXT/Projects/ocr_datasets/alpr/recogn/latin/websites/www.platesmania.com/si/",
	"/media/mamadou/TOSHIBA EXT/Projects/ocr_datasets/alpr/recogn/latin/websites/www.platesmania.com/sk/",
	"/media/mamadou/TOSHIBA EXT/Projects/ocr_datasets/alpr/recogn/latin/websites/www.platesmania.com/su/",
	"/media/mamadou/TOSHIBA EXT/Projects/ocr_datasets/alpr/recogn/latin/websites/www.platesmania.com/ua/",
	"/media/mamadou/TOSHIBA EXT/Projects/ocr_datasets/alpr/recogn/latin/websites/www.platesmania.com/uk/",
	"/media/mamadou/TOSHIBA EXT/Projects/ocr_datasets/alpr/recogn/latin/websites/www.platesmania.com/us/",
	"/media/mamadou/TOSHIBA EXT/Projects/ocr_datasets/alpr/recogn/latin/websites/www.platesmania.com/uz/",
	"/media/mamadou/TOSHIBA EXT/Projects/ocr_datasets/alpr/recogn/latin/websites/www.platesmania.com/xx/",
]

if __name__ == '__main__':
    # Configure logger
    coloredlogs.install(level='INFO')
    logger = logging.getLogger('org.doubango.dataset')
    logger.info('Building dataset...')
        
    envs = {}
    cnts = {}
    for target in ['train', 'val']:
        path = os.path.join(DEST_FOLDER, '{}.lmbd'.format(target))
        if os.path.exists(path):
            shutil.rmtree(path)

        envs[target] = lmdb.open(path, map_size=int(CACHE_SIZE))
        cnts[target] = 1
    
    val_pop = list(np.arange(0, 1, VAL_RATIO))
    
    back_img = Image.new('RGB', (150, 150))
    
    for i, folder in enumerate(FOLDERS):
        logger.info('Processing [%d/%d]...', i, len(FOLDERS))
        files = [os.path.join(folder, file) for file in os.listdir(folder) if os.path.isfile(os.path.join(folder, file))]
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
            annotation = dom.find('annotation')
            
            if not annotation or len(annotation) == 0:
                logger.error('{} has no annotation'.format(xml_file))
                continue
            
            object = annotation.find('object')
            if not object or len(object) == 0:
                logger.error('{} has no object'.format(xml_file))
                continue
            
            text = object.find('text')
            if not text or len(text) == 0:
                logger.error('{} has no text'.format(xml_file))
                continue
            license_numer = text.text
            
            target = 'val' if (random.choice(val_pop) == 0) else 'train'
            
            image = Image.open(file).convert('RGB')
            if image.size[0] != 300 or image.size[1] != 150:
                logger.error('{} has invalid size ({})'.format(file, image.size))
                continue
            
            image = image.crop((0, 0, 150, 150))
            diff = ImageChops.difference(image, back_img)
            diff = ImageChops.add(diff, diff, 2.0, -20)
            bbox = diff.getbbox()
            if bbox:
                image = image.crop(bbox)
                
            buffer = io.BytesIO()
            image.save(buffer, 'JPEG')
            buffer.seek(0)
                
            cache = {}
            imageKey = 'image-%09d' % cnts[target]
            labelKey = 'label-%09d' % cnts[target]
            cache[imageKey] = buffer.read()
            cache[labelKey] = license_numer.encode()
            with envs[target].begin(write=True) as txn:
                for k, v in cache.items():
                    txn.put(str(k).encode(), v)
            
            cnts[target] += 1
            

        #break
            
    
    # Write annotations
    for target in envs.keys():
        cache = {
            "num-samples": str(cnts[target] - 1).encode()
        }
        with envs[target].begin(write=True) as txn:
            for k, v in cache.items():
                txn.put(str(k).encode(), v)
        
        envs[target].close()
    
    logger.info('!!DONE!!')
    
                
            
    
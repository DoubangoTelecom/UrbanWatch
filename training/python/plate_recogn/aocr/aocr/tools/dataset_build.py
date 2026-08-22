
import os, random, numpy as np, io, shutil, argparse
from bs4 import BeautifulSoup
from PIL import Image, ImageChops

IMAGE_EXTs = ['tif','png','jpg','jpeg','bmp']

# latin: # Comes from https://github.com/DoubangoTelecom/ultimateALPR/blob/6832b3b1e5e35942723c1b32a6aef751ddf861e2/training/recogn_copy_files.cxx#L61
FOLDERS = {
    "latin": [
            ## Collected ##
            "collected/000/",
            "collected/001/",

            ## From customers ##
            "mexico/seguritech/imagenes_carros/",
            "siri_lanka/elogiclanka/",
            "turkey/noveltybilisim/",
            "new_zeland/nick_bolton/",
            "taiwan/acerits/",
            "taiwan/avadesign/",
            "usa/skycopinc-tennessee/",
            "usa/stacked/",
            "indonesia/",
            "myriade/",

            ## Scrapping ##
            "scrap/platesmania/brazil/",
            "scrap/platesmania/canada/",
            "scrap/platesmania/usa/",
            "scrap/platesmania/kazakhstan/",
            "scrap/platesmania/turkey/",

            ## Google images ##
            "google images/",
            "siri_lanka/googleimage/",

            ## Extrats ##
            "extras/",

            ## Inverse (Ooops) ##
            "inverse/",
            
            ## Cities ##
            "cities/london/",
            
            ## Changed from png to jpg ##
            "png_to_jpg/",

            ## Croatia ##
            "croatia/",

            ## Greece ##
            "greece/difficult_dirt_shadows/",
            "greece/difficult_shadows/",
            "greece/difficult_tracks_night/",
        
            ## Brazil ##
            "brazil/day/",
            "brazil/night/",

            ## India ##
            "india/",

            ## Tunisia ##
            "tunisia/",
            
            ## Kaggle ##
            "kaggle/training_set/",

            ## Old ##
            "old/",

            ## OpenALPR ##
            "openalpr/br/",
            "openalpr/eu/",
            "openalpr/us/",

            ## Google Search ##
            "websites/www.google.com - search/",
            "websites/www.google.com - search - us - mosaic/",

            ## alpr-lpci-google-images-download ##
            "alpr-lpci-google-images-download/United_Arab_Emirates-Abu_Dhabi",
            "alpr-lpci-google-images-download/United_Arab_Emirates-Dubai",
            "alpr-lpci-google-images-download/Kyrgyzstan",
            "alpr-lpci-google-images-download/Kazakhstan",
            "alpr-lpci-google-images-download/Uzbekistan",
            "alpr-lpci-google-images-download/Turkmenistan",
            "alpr-lpci-google-images-download/Tajikistan",
            "alpr-lpci-google-images-download/Spain",
            "alpr-lpci-google-images-download/USA-Florida",
            "alpr-lpci-google-images-download/USA-Utah",
            "alpr-lpci-google-images-download/USA-California",
            "alpr-lpci-google-images-download/Russia",
            "alpr-lpci-google-images-download/Brasil",

        #	if 0 // Not done yet
            #"alpr-lpci-google-images-download/United_Arab_Emirates-Dubai",
            #"alpr-lpci-google-images-download/United_Arab_Emirates-Ajman",
            #"alpr-lpci-google-images-download/United_Arab_Emirates-Fujairah",
            #"alpr-lpci-google-images-download/United_Arab_Emirates-Ras_al_Khaimah",
            #"alpr-lpci-google-images-download/United_Arab_Emirates-Sharjah",
            #"alpr-lpci-google-images-download/United_Arab_Emirates-Umm_al_Qaiwain",
        #	endif

            ## olavsplates ##
            "websites/www.olavsplates.com/000/",
            "websites/www.olavsplates.com/001/",
            "websites/www.olavsplates.com/002/",
            "websites/www.olavsplates.com/003/",
            "websites/www.olavsplates.com/004/",
            "websites/www.olavsplates.com/005/",
            "websites/www.olavsplates.com/006/",
            "websites/www.olavsplates.com/007/",

            ## vizura.net ##
            "vizura.net/TrainingSet1/000/",
            "vizura.net/TrainingSet1/001/",
            "vizura.net/TrainingSet1/002/",
            "vizura.net/TrainingSet1/003/",
            "vizura.net/TrainingSet1/004/",
            "vizura.net/TrainingSet1/005/",
            "vizura.net/TrainingSet1/006/",
            "vizura.net/TrainingSet1/007/",
            "vizura.net/TrainingSet1/008/",
            "vizura.net/TrainingSet1/009/",

            ## platesmania ##
            "websites/www.platesmania.com/ae/",
            "websites/www.platesmania.com/at/",
            "websites/www.platesmania.com/az/",
            "websites/www.platesmania.com/be/",
            "websites/www.platesmania.com/bg/",
            "websites/www.platesmania.com/ca/",
            "websites/www.platesmania.com/ch/",
            "websites/www.platesmania.com/de/",
            "websites/www.platesmania.com/dk/",
            "websites/www.platesmania.com/fi/",
            "websites/www.platesmania.com/fr/",
            "websites/www.platesmania.com/gr/",
            "websites/www.platesmania.com/hr/",
            "websites/www.platesmania.com/il/",
            "websites/www.platesmania.com/it/",
            "websites/www.platesmania.com/kz/",
            "websites/www.platesmania.com/lu/",
            "websites/www.platesmania.com/mc/",
            "websites/www.platesmania.com/me/",
            "websites/www.platesmania.com/ml/",
            "websites/www.platesmania.com/mx/",
            "websites/www.platesmania.com/no/",
            "websites/www.platesmania.com/pl/",
            "websites/www.platesmania.com/pt/",
            "websites/www.platesmania.com/ro/",
            "websites/www.platesmania.com/rs/",
            "websites/www.platesmania.com/ru/",
            "websites/www.platesmania.com/se/",
            "websites/www.platesmania.com/si/",
            "websites/www.platesmania.com/sk/",
            "websites/www.platesmania.com/su/",
            "websites/www.platesmania.com/ua/",
            "websites/www.platesmania.com/uk/",
            "websites/www.platesmania.com/us/",
            "websites/www.platesmania.com/uz/",
            "websites/www.platesmania.com/xx/",

    ] # end-of-"latin"



} # end-of-FOLDER

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--target', type=str, required=True, help="target type. Must be latin, korean, chinese or brazilian")
    parser.add_argument('--in_folder', type=str, required=True, help="input folder")
    parser.add_argument('--out_folder', type=str, required=True, help="output folder")
    parser.add_argument('--val_ratio', type=float, required=False, default=0.1, help="validation ration. Within ]0,1[]")
    parser.add_argument("--legacy_img_size", type=int, required=False, default=150, help="legacy image size")
    opts = parser.parse_args()
    
    '''opts = type('obj', (object,), {
        'target': 'latin',
        'in_folder': '/media/mamadou/TOSHIBA EXT/Projects/ocr_datasets/alpr/recogn/latin',
        'out_folder': '/home/projects/urban-watch/plate_recogn/aocr/datasets/latin',
        'val_ratio': 0.1,
        'legacy_img_size': 150
    })
    opts = type('obj', (object,), {
        'target': 'latin',
        'in_folder': 'E:/Projects/ocr_datasets/alpr/recogn/latin',
        'out_folder': 'C:/Users/dmi83/Downloads/test_aocr_datasets',
        'val_ratio': 0.1,
        'legacy_img_size': 150
    })'''
    
    # Sanity check
    assert opts.val_ratio < 1.0 and opts.val_ratio > 0.0, f'Invalid validation ration: {opts.val_ratio}'

    return opts

if __name__ == '__main__':
    opts = parse_args()

    print('Building dataset...')

    # Create dest folders
    for folder in ['train', 'val']:
        path = os.path.join(opts.out_folder, f'imgs_{folder}')
        if os.path.isdir(path):
            shutil.rmtree(path)
        os.makedirs(path, exist_ok=True)
    
    # Moving files
    anno_list = {
        'train': { 'imgs': [], 'texts': [] },
        'val': { 'imgs': [], 'texts': [] }
    }
    in_folders = FOLDERS[opts.target]
    back_img = Image.new('RGB', (opts.legacy_img_size, opts.legacy_img_size))
    for i, folder in enumerate(in_folders):
        print(f'[{i:3d}/{len(in_folders):3d}] Processing [{folder}]...')           
        
        files = [os.path.join(opts.in_folder, folder, name) for name in os.listdir(os.path.join(opts.in_folder, folder)) if name.split('.')[-1].lower() in IMAGE_EXTs]
        num_files = len(files)
        if num_files == 0:
            print(f'No zero files in {folder}')
            continue
        random.Random(1983).shuffle(files)
        num_val_files = 0 if num_files == 1 else max(1, int(num_files * opts.val_ratio))
        for j, file in enumerate(files):
            try:
                fparts = file.split('.')
                ext = fparts[-1].lower()
                if not ext in IMAGE_EXTs:
                    continue
                
                xml_file = '.'.join(fparts[:-1]) + '.xml'
                if not os.path.isfile(xml_file):
                    print('{} do not exist'.format(xml_file))
                    continue
                
                dom = BeautifulSoup(io.open(xml_file, mode="r", encoding="utf-8").read(), features='xml')
                annotation = dom.find('annotation')
                
                if not annotation or len(annotation) == 0:
                    print('{} has no annotation'.format(xml_file))
                    continue
                
                object = annotation.find('object')
                if not object or len(object) == 0:
                    print('{} has no object'.format(xml_file))
                    continue
                
                text = object.find('text')
                if not text or len(text) == 0:
                    print('{} has no text'.format(xml_file))
                    continue
                
                target = 'val' if j < num_val_files else 'train'
                
                image = Image.open(file).convert('RGB')
                if (image.size[0] % opts.legacy_img_size) != 0 or image.size[1] != opts.legacy_img_size:
                    print('{} has invalid size ({})'.format(file, image.size))
                    continue
                
                image = image.crop((0, 0, opts.legacy_img_size, opts.legacy_img_size))
                diff = ImageChops.difference(image, back_img)
                diff = ImageChops.add(diff, diff, 2.0, -20)
                bbox = diff.getbbox()
                if bbox:
                    image = image.crop(bbox)                
                
                out_file_name = f'{i}_{j}.png' # "png" is faster to encode/decode than "jpg"
                out_file_path = os.path.join(opts.out_folder, f'imgs_{target}', out_file_name)
                assert not os.path.exists(out_file_path), f'File at {out_file_path} already exists'
                image.save(out_file_path)

                anno_list[target]['imgs'].append(out_file_name)
                anno_list[target]['texts'].append(text.text)

            except Exception as error:
                print(f'An exception occurred: {error}. File: {file}')
                continue
                 
    
    # Writing list of files (imgs.txt)
    # This will make it faster to load the files without listing the directory
    # for each run
    # Also write the plate numbers
    for target in ['train', 'val']:
        for elt in ['imgs', 'texts']:
            with open(os.path.join(opts.out_folder, f'imgs_{target}', f'{elt}.txt'), "w") as f:
                f.write('\n'.join(anno_list[target][elt]))

                
    print('!!! DONE !!!')
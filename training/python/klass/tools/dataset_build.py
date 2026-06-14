import argparse, os, coloredlogs, logging, random, numpy as np, io, shutil
from PIL import Image, ImageOps

IMAGE_EXTs = ['tif','png','jpg','jpeg','bmp']
TARGETS = ['vcr', 'lpci', 'vbsr', 'vmmr']

# https://github.com/DoubangoTelecom/ultimateALPR/blob/482d602656c632258b86cc955457375a7624b8be/training/klass_vmmr_fuse.cxx#L23
VMMR_USELESS_WORDS = "i,ii,iii,iv,v,vi,vii,viii,sp,sp2,pickup,coupe,sportsvan,wagon,estate,convertible,avant,hatchback,classic,offroad,alltrack,variant,srt,aircross,s,cross,crossove,tourer,cc,ct,sw,break,electric,cabriolet,sedan,convertable,classic,solara,xle,xl,plus,speedster,mk8,r36,vista,quattro,mach,sport,sports,touring,country,srt,active,tourer,rubicon,sahara,tj,unlimited,pinin,tino,tepee,combi,urmodell,grand,tour,freetrack,+,plus,verso,h1,superleggera,gt,euv,gtb,e-tech,".split(',')
VMMR_ALPHABET = set("abcdefghijklmnopqrstuvwxyz0123456789 -_&!+#") # chars allowed in the name
YEAR_STEP = int(3000)

def vmmr_parse(name :str) -> object:
    """ https://github.com/DoubangoTelecom/ultimateALPR/blob/482d602656c632258b86cc955457375a7624b8be/training/klass_vmmr_fuse.cxx#L128 """
    
    # {name} could be "renault_megane_grand_tour_2003"
    # {name} format is make_model_year
    parts = name.split('_')
    assert len(parts) >= 3, f'Invalid name {name}'
    
    return type('obj', (object,), {
        'make': parts[0],
        'model': " ".join(parts[1:len(parts)-1]),
        'year': parts[-1]
    })
    
def vmmr_remap_make(make :str) -> str:
    """ https://github.com/DoubangoTelecom/ultimateALPR/blob/482d602656c632258b86cc955457375a7624b8be/training/klass_vmmr_fuse.cxx#L182 """
    # Opel Astra, Corsa, Antara... are the same as vauxhall Astra, Corsa, Antara...
    if make == "vauxhall": return "opel"
    return make

def vmmr_remap_model(model :str) -> str:
    """ https://github.com/DoubangoTelecom/ultimateALPR/blob/482d602656c632258b86cc955457375a7624b8be/training/klass_vmmr_fuse.cxx#L158 """
    
    model = model.replace('-', ' ')
    
    parts = model.split(' ')
    cleans = []
    for part in parts:
        if not part in VMMR_USELESS_WORDS:
            cleans.append(part)
    if len(cleans) == 0:
        print(f'{model} is invalid model name or ill-formed')
        return model
    return ' '.join(cleans)

def vmmr_remap_year(year :int) -> int:
    """ https://github.com/DoubangoTelecom/ultimateALPR/blob/482d602656c632258b86cc955457375a7624b8be/training/klass_vmmr_fuse.cxx#L21 """
    '''
        The fusing is based on the year step to reduce the number of classes.
        fused = (original/step) * step, using integer operations
        For example, let's say the year is 1987 and the step is 10:
            - fused = (1987/10)*10 = 1980
            - For display, say year is "1980-(1980+(step-1))"
    '''
    #define YEAR_STEP				3000
    #define YEAR_FUSE(__Year__)		((static_cast<int>(__Year__) / YEAR_STEP)*YEAR_STEP)
    return int((int(year) // YEAR_STEP)*YEAR_STEP)
    
def vmmr_fuse(name :str) -> str:
    """ https://github.com/DoubangoTelecom/ultimateALPR/blob/482d602656c632258b86cc955457375a7624b8be/training/klass_vmmr_fuse.cxx#L189 """
    
    # Make sur name is valid
    for c in name: 
        assert c in VMMR_ALPHABET, f'{c} not in alphabet'
    
    # Parse the name and extract 'make', 'model' and 'year'
    mmy = vmmr_parse(name)
    
    # Parse year
    year_str = mmy.year.split('-')[0] # '2026' or '2010-2015'
    assert year_str.isdigit(), f'{mmy.year} not a integer'
    yearInt = int(year_str)
    
    return f'{vmmr_remap_make(mmy.make)}_{vmmr_remap_model(mmy.model)}_{vmmr_remap_year(yearInt):04d}'

def process(opt):
    """ Processing """    
    coloredlogs.install(level='INFO')
    logger = logging.getLogger('org.doubango.dataset')
    logger.info('Building dataset...')
    
    # List folders
    logger.info('Listing folders...')
    folders = [name for name in os.listdir(opt.in_folder) if not os.path.isfile(os.path.join(opt.in_folder, name))]
    assert len(folders) > 0, f'Empty number of folders in {opt.in_folder}'
    random.Random(1983).shuffle(folders) # randomize folders
    
    # Create dest folders
    for folder in ['train', 'val']:
        path = os.path.join(opt.out_folder, f'imgs_{folder}')
        if os.path.isdir(path):
            shutil.rmtree(path)
        os.makedirs(path, exist_ok=True)
        
    # Moving files
    klass_list = []
    img_list = {
        'train': [],
        'val': []
    }
    for i, folder in enumerate(folders):
        logger.info(f'[{i:3d}/{len(folders):3d}] Processing [{folder}]...')
        # Klass fusion
        klass_name = vmmr_fuse(folder) if opt.target == 'vmmr' else folder
        klass_index = klass_list.index(klass_name) if klass_name in klass_list else -1
        if klass_index < 0:
            klass_index = len(klass_list)
            klass_list.append(klass_name)            
        
        files = [name for name in os.listdir(os.path.join(opt.in_folder, folder)) if name.split('.')[-1].lower() in IMAGE_EXTs]
        num_files = len(files)
        assert num_files > 0, f'Few number of files ({num_files}) in {folder}'
        random.Random(1983).shuffle(files)
        num_val_files = 0 if num_files == 1 else max(1, int(num_files * opt.val_ratio))
        for j, file in enumerate(files):
            try:
                path = os.path.join(opt.in_folder, folder, file)
                image = Image.open(path).convert('RGB')
                target = 'val' if j < num_val_files else 'train'
                out_file_name = f'{klass_index}_{j}_{folder}.jpg' # format = '#class#_#id#_name.jpg'
                out_file_path = os.path.join(opt.out_folder, f'imgs_{target}', out_file_name)
                image = ImageOps.contain(image, (opt.img_w, opt.img_h))
                image.save(out_file_path)
            except Exception as error:
                logger.error(f'An exception occurred: {error}. File: {file}')
                continue
                
            img_list[target] += [out_file_name, ]
            
    # Write classes
    logger.info('Writting classes...')
    with open(os.path.join(opt.out_folder, "classes.txt"), "w") as f:
        f.write('\n'.join(klass_list))        
    
    # Writing list of files (list.txt)
    # This will make it faster to load the files without listing the directory
    # for each run
    for folder in ['train', 'val']:
        with open(os.path.join(opt.out_folder, f'imgs_{folder}', 'list.txt'), "w") as f:
            f.write('\n'.join(img_list[folder]))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--target', type=str, required=True, help="target type. Must be vcr, vbsr, lpci or vmmr")
    parser.add_argument('--in_folder', type=str, required=True, help="input folder")
    parser.add_argument('--out_folder', type=str, required=True, help="output folder")
    parser.add_argument('--val_ratio', type=float, required=False, default=0.1, help="validation ration. Within ]0,1[]")
    parser.add_argument("--img_w", type=int, required=True, help="target image width")
    parser.add_argument("--img_h", type=int, required=True, help="target image height")
    opt = parser.parse_args()
    
    '''opt = type('obj', (object,), {
        'target': 'vcr', # vcr, lpci, vmmr...
        'in_folder': 'E:/Projects/ocr_datasets/alpr-vcr-untouched/train',
        'out_folder': 'E:/Projects/ocr_datasets/garbage/vcr',
        'val_ratio': 0.1,
        'img_w': 128,
        'img_h': 128
    })
    opt = type('obj', (object,), {
        'target': 'vmmr', # vcr, lpci, vmmr...
        'in_folder': 'E:/Projects/ocr_datasets/alpr-vmmr-untouched/train',
        'out_folder': 'E:/Projects/ocr_datasets/garbage/vmmr',
        'val_ratio': 0.1,
        'img_w': 128,
        'img_h': 128
    })'''
    
    # Sanity check
    assert opt.target in TARGETS, f'Invalid taget: {opt.target}'
    assert opt.val_ratio < 1.0 and opt.val_ratio > 0.0, f'Invalid validation ration: {opt.val_ratio}'

    # Processing
    process(opt)
    
    print('!!! DONE !!!')
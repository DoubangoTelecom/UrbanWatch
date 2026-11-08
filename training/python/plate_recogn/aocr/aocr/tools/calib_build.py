import os, argparse, random, shutil, gzip, numpy as np
from PIL import Image, ImageOps
from aocr.config import Config

def pre_process(img: Image, cfg: Config, channel_first :bool=True):
    # Normalization
    mean = np.array(cfg.model.normalize[0], dtype=float).reshape([1, 1, 3])
    std = np.array(cfg.model.normalize[1], dtype=float).reshape([1, 1, 3])
    img = ((np.array(img).astype(np.float32) - mean) / std).astype(np.float32)
    
    # Transpose
    if channel_first:
        img = img.transpose((2, 0, 1)) # nhwc ->  nchw
    return img[None,...]

def get_image_list(path, image_ext = [".jpg", ".jpeg", ".webp", ".bmp", ".png"]):
    image_names = []
    for maindir, subdir, file_name_list in os.walk(path):
        for filename in file_name_list:
            apath = os.path.join(maindir, filename)
            ext = os.path.splitext(apath)[1]
            if ext in image_ext:
                image_names.append(apath)
    return image_names

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Create calibartion dataset",
    )
    parser.add_argument('--cfg_path', required=True, help="path to config file")
    parser.add_argument('--in_folder', type=str, required=True, help="input folder")
    parser.add_argument("--out_folder", type=str, required=True, help="Output folder")
    parser.add_argument("--num_images", type=int, required=False, default=15000, help="Image size")
    args = parser.parse_args()

    # Parse config
    cfg = Config.parse(args.cfg_path)
    
    # Re-create the output folder
    if os.path.exists(args.out_folder):
        shutil.rmtree(args.out_folder)
    os.makedirs(args.out_folder)
        
    # Create images (RKNN requires images with the exact size)
    paths = get_image_list(args.in_folder)
    assert len(paths) > 0, "List of images is empty"
    random.shuffle(paths)
    if len(paths) > args.num_images:
        paths = paths[:args.num_images]

    dataset_paths = os.path.join(args.out_folder, 'dataset.txt') # for RKNN
    dataset_hailo_folder = os.path.join(args.out_folder, 'hailo') # for Hailo
    os.makedirs(dataset_hailo_folder)
    target_size = (cfg.model.imgW, cfg.model.imgH)
    with open(dataset_paths, "w") as paths_file, gzip.open(os.path.join(args.out_folder, 'dataset.npy.gz'), 'wb') as nvidia_modelopt_file:
        for i, path in enumerate(paths):
            # Print progression
            print('[{:3d}/{:3d}] Processing {}...'.format(i, len(paths), path))

            # Read image
            image = Image.open(path).convert('RGB')
            
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
                
            # Write to disk
            image_name = os.path.basename(path).replace(' ', '_') # rknn doesn't support space in names
            dst_path = os.path.join(args.out_folder, image_name)
            img.save(dst_path)
            np.save(f"{dst_path.split('.')[0]}.npy", np.asarray(img).astype(np.float32))
            np.save(nvidia_modelopt_file, pre_process(img, cfg))
            paths_file.write(f"{image_name}\n")

        
    print('!!! DONE !!!')
    
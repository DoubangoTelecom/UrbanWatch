import os, argparse, random, shutil, numpy as np
from PIL import Image, ImageOps

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
    parser.add_argument("--padding", type=str, required=True, help="Whether to pad the image to keep aspect ratio")
    parser.add_argument("--grayscale", type=str, required=True, help="RGB or grayscale")
    parser.add_argument('--in_folder', type=str, required=True, help="input folder")
    parser.add_argument("--out_folder", type=str, required=True, help="Output folder")
    parser.add_argument("--image_size", type=int, required=True, help="Image size")
    parser.add_argument("--num_images", type=int, required=False, default=15000, help="Image size")
    args = parser.parse_args()
    
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
    target_size = (args.image_size, args.image_size)
    with open(dataset_paths, "w") as f:
        for i, path in enumerate(paths):
            # Print progression
            print('[{:3d}/{:3d}] Processing {}...'.format(i, len(paths), path))

            # Read image
            image = Image.open(path).convert('RGB')
            
            # Resize
            if args.padding == 'True':
                tmp = ImageOps.contain(image, target_size)
                img = Image.new(tmp.mode, target_size, 0)
                img.paste(tmp, (0, 0))
            else:
                img = image.resize(target_size)
                
            # Grayscale or RGB
            if args.grayscale == "True":
                img = img.convert('L')
                
            # Write to disk
            image_name = os.path.basename(path)
            dst_path = os.path.join(args.out_folder, image_name)
            img.save(dst_path)
            np.save(f"{dst_path.split('.')[0]}.npy", np.asarray(img).astype(np.float32))
            f.write(f"{image_name}\n")
        
    print('!!! DONE !!!')
    
import os, argparse, random, shutil, numpy as np
from export_utils import get_image_list
from PIL import Image, ImageOps

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Create calibartion dataset",
    )
    parser.add_argument("--in_folder", type=str, required=True, help="Input folder")
    parser.add_argument("--out_folder", type=str, required=True, help="Output folder")
    parser.add_argument("--image_size", type=int, required=True, help="Image size")
    parser.add_argument("--kar", action=argparse.BooleanOptionalAction, required=True, help="Keep Aspect Ration")
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
    with open(dataset_paths, "w") as f:
        for i, path in enumerate(paths):
            print('[{:3d}/{:3d}] Processing {}...'.format(i, len(paths), path))
            dst_path = os.path.join(args.out_folder, os.path.basename(path))
            image = Image.open(path).convert('RGB')
            if args.kar:
                tmp = ImageOps.contain(image, (args.image_size, args.image_size), Image.BILINEAR)
                image = Image.new(tmp.mode, (args.image_size, args.image_size), 0)
                image.paste(tmp, (0, 0))
            else:
                image = image.resize((args.image_size, args.image_size), Image.BILINEAR)
            image.save(dst_path)
            np.save(f"{dst_path.split('.')[0]}.npy", np.asarray(image).astype(np.float32))
            f.write(f"{os.path.basename(path)}\n")

        
    print('!!! DONE !!!')
    
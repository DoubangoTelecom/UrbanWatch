import os, argparse, random, shutil, numpy as np, six, lmdb
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
    parser.add_argument("--lmdb_database", type=str, required=True, help="Path to lmdb database")
    parser.add_argument("--out_folder", type=str, required=True, help="Output folder")
    parser.add_argument("--image_size", type=int, required=True, help="Image size")
    parser.add_argument("--num_images", type=int, required=False, default=15000, help="Image size")
    args = parser.parse_args()
    
    # Re-create the output folder
    if os.path.exists(args.out_folder):
        shutil.rmtree(args.out_folder)
    os.makedirs(args.out_folder)
        
    # Open database
    lmdb_env = lmdb.open(args.lmdb_database, max_readers=32, readonly=True, lock=False, readahead=False, meminit=False)
    if not lmdb_env:
        print('cannot create lmdb from %s' % (args.lmdb_database))
        exit(0)
        
    # Get list of images ids
    with lmdb_env.begin(write=False) as txn:
        lmdb_nSamples = min(int(txn.get('num-samples'.encode())), args.num_images)
        images_ids = [index + 1 for index in range(lmdb_nSamples)]
        random.shuffle(images_ids)
    
    dataset_paths = os.path.join(args.out_folder, 'dataset.txt') # for RKNN
    dataset_hailo_folder = os.path.join(args.out_folder, 'hailo') # for Hailo
    os.makedirs(dataset_hailo_folder)
    target_size = (args.image_size, args.image_size)
    with open(dataset_paths, "w") as f:
        for i, image_id in enumerate(images_ids):
            # Print progression
            print('[{:3d}/{:3d}] Processing {}...'.format(i, len(images_ids), image_id))
            # Read image from lmdb database
            with lmdb_env.begin(write=False) as txn:
                img_key = f'image-%09d'.encode() % image_id
                imgbuf = txn.get(img_key)

            buf = six.BytesIO()
            buf.write(imgbuf)
            buf.seek(0)
            image = Image.open(buf).convert('RGB')
            
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
            image_name = f'image-%09d' % image_id + '.jpg'
            dst_path = os.path.join(args.out_folder, image_name)
            img.save(dst_path)
            np.save(f"{dst_path.split('.')[0]}.npy", np.asarray(img).astype(np.float32))
            f.write(f"{image_name}\n")
        
    lmdb_env.close()
        
    print('!!! DONE !!!')
    
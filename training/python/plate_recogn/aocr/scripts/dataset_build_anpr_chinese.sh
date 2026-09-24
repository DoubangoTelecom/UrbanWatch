# mount /dev/sda2 "/media/mamadou/TOSHIBA EXT"
python aocr/tools/dataset_build.py \
--target 'chinese' \
--in_folder '/media/mamadou/TOSHIBA EXT/Projects/ocr_datasets/alpr/recogn/china' \
--out_folder '/home/projects/urban-watch/plate_recogn/aocr/datasets/chinese' \
--val_ratio 0.1
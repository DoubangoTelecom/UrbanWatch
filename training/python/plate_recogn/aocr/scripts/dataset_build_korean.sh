# mount /dev/sda2 "/media/mamadou/TOSHIBA EXT"
python aocr/tools/dataset_build.py \
--target 'korean' \
--in_folder '/media/mamadou/TOSHIBA EXT/Projects/ocr_datasets/alpr/recogn/korea' \
--out_folder '/home/projects/urban-watch/plate_recogn/aocr/datasets/korean' \
--val_ratio 0.1
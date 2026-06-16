python aocr/tools/calib_build.py \
--num_images 20000 \
--padding False \
--grayscale False \
--image_size 128 \
--lmdb_database datasets/latin/train.lmbd \
--out_folder datasets/latin/calibration
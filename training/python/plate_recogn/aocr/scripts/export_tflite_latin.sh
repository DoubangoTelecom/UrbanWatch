CUDA_VISIBLE_DEVICES=-1 python aocr/tools/export_tflite.py \
--config configs/config_latin.yml \
--weights saved_models/anpr_latin/best_norm_ED.pth \
--out_path aocr_anpr_latin.tflite \
--per_channel "False" \
--int16_activation "False" \
--calibration_dataset datasets/latin/calibration/dataset.txt
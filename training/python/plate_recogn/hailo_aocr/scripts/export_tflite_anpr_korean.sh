CUDA_VISIBLE_DEVICES=-1 python aocr/tools/export_tflite.py \
--config configs/config_anpr_korean.yml \
--weights saved_models/anpr_korean/best_norm_ED.pth \
--out_path aocr_anpr_korean.tflite \
--per_channel "False" \
--int16_activation "False" \
--calibration_dataset datasets/korean/calibration/dataset.txt
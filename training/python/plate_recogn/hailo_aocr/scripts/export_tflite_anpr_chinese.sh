CUDA_VISIBLE_DEVICES=-1 python aocr/tools/export_tflite.py \
--config configs/config_anpr_chinese.yml \
--weights saved_models/anpr_chinese/best_norm_ED.pth \
--out_path aocr_anpr_chinese.tflite \
--per_channel "False" \
--int16_activation "False" \
--calibration_dataset datasets/chinese/calibration/dataset.txt
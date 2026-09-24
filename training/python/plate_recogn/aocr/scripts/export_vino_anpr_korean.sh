CUDA_VISIBLE_DEVICES=-1 python aocr/tools/export_vino.py \
--out_path vino_models/aocr_anpr_korean.vino/model.xml \
--cfg_path configs/config_anpr_korean.yml \
--weights saved_models/anpr_korean/best_norm_ED.pth \
--calibration_dataset datasets/korean/calibration/dataset.txt

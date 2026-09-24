CUDA_VISIBLE_DEVICES=-1 python aocr/tools/export_vino.py \
--out_path vino_models/aocr_anpr_chinese.vino/model.xml \
--cfg_path configs/config_anpr_chinese.yml \
--weights saved_models/anpr_chinese/best_norm_ED.pth \
--calibration_dataset datasets/chinese/calibration/dataset.txt

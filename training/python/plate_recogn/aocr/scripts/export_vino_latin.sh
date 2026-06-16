CUDA_VISIBLE_DEVICES=-1 python aocr/tools/export_vino.py \
--out_path vino_models/aocr_anpr_latin.vino/model.xml \
--cfg_path configs/config_latin.yml \
--weights saved_models/anpr_latin/best_norm_ED.pth \
--calibration_dataset datasets/latin/calibration/dataset.txt

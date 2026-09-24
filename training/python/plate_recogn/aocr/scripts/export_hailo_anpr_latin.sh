CUDA_VISIBLE_DEVICES=-1 python aocr/tools/export_hailo.py \
    --platforms hailo8,hailo8l,hailo8r \
    --target anpr_latin \
    --out_folder hailo_models \
    --calibration_dataset datasets/latin/calibration \
    --cfg_path configs/config_anpr_latin.yml \
    --model_path saved_models/anpr_latin/best_norm_ED.pth
    
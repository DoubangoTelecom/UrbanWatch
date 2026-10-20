CUDA_VISIBLE_DEVICES=-1 python aocr/tools/export_hailo.py \
    --platforms hailo8,hailo8l,hailo8r \
    --target anpr_chinese \
    --out_folder hailo_models \
    --calibration_dataset datasets/chinese/calibration \
    --cfg_path configs/config_anpr_chinese.yml \
    --model_path saved_models/anpr_chinese/best_norm_ED.pth
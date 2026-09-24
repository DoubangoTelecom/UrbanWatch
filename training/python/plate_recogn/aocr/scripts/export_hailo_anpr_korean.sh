CUDA_VISIBLE_DEVICES=-1 python aocr/tools/export_hailo.py \
    --platforms hailo8,hailo8l,hailo8r \
    --target anpr_korean \
    --out_folder hailo_models \
    --calibration_dataset datasets/korean/calibration \
    --cfg_path configs/config_anpr_korean.yml \
    --model_path saved_models/anpr_korean/best_norm_ED.pth
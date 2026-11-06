CUDA_VISIBLE_DEVICES=-1 python tools/export_hailo.py \
    --platforms hailo8,hailo8l,hailo8r \
    --out_folder hailo_models \
    --calibration_dataset datasets/vmmr/calibration \
    --cfg_path configs/vmmr.yml \
    --model_path saved_models/vmmr.pth
    
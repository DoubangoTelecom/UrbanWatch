CUDA_VISIBLE_DEVICES=-1 python tools/export_hailo.py \
    --platforms hailo8,hailo8l,hailo8r \
    --out_folder hailo_models \
    --calibration_dataset datasets/vcr/calibration \
    --cfg_path configs/vcr.yml \
    --model_path saved_models/vcr.pth
    
CUDA_VISIBLE_DEVICES=-1 python tools/export_hailo.py \
    --platforms hailo8,hailo8l,hailo8r \
    --target main_desktop \
    --out_folder hailo_models \
    --calibration_dataset datasets/anpr_main/calibration_desktop \
    --cfg_path config/anpr_desktop.yml \
    --model_path workspace/anpr_desktop/model_best/model_best.ckpt
    
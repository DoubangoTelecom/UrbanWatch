CUDA_VISIBLE_DEVICES=-1 python tools/export_hailo.py \
    --platforms hailo8,hailo8l,hailo8r \
    --target main_mobile \
    --out_folder hailo_models \
    --calibration_dataset datasets/anpr_main/calibration \
    --cfg_path config/anpr_mobile.yml \
    --model_path workspace/anpr_mobile/model_best/model_best.ckpt

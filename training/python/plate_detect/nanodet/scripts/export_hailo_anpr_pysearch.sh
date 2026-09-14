CUDA_VISIBLE_DEVICES=-1 python tools/export_hailo.py \
    --platforms hailo8,hailo8l,hailo8r \
    --target pysearch \
    --out_folder hailo_models \
    --calibration_dataset datasets/anpr_pysearch/calibration \
    --cfg_path config/anpr_pysearch.yml \
    --model_path workspace/anpr_pysearch/model_best/model_best.ckpt

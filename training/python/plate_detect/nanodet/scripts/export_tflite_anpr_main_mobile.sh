CUDA_VISIBLE_DEVICES=-1 python tools/export_tflite.py \
    --per_channel False \
    --out_path anpr_detect_main_mobile.tflite \
    --cfg_path config/anpr_mobile.yml \
    --calibration_dataset datasets/anpr_main/calibration/dataset.txt \
    --model_path workspace/anpr_mobile/model_best/model_best.ckpt

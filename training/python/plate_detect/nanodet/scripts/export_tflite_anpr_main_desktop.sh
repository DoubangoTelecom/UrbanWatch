CUDA_VISIBLE_DEVICES=-1 python tools/export_tflite.py \
    --per_channel False \
    --out_path anpr_detect_main_desktop.tflite \
    --cfg_path config/anpr_desktop.yml \
    --calibration_dataset datasets/anpr_main/calibration/dataset.txt \
    --model_path workspace/anpr_desktop/model_best/model_best.ckpt

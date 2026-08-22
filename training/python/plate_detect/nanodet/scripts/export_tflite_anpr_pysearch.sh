CUDA_VISIBLE_DEVICES=-1 python tools/export_tflite.py \
    --per_channel False \
    --out_path anpr_detect_pysearch.tflite \
    --cfg_path config/anpr_pysearch.yml \
    --calibration_dataset datasets/anpr_pysearch/calibration/dataset.txt \
    --model_path workspace/anpr_pysearch/model_best/model_best.ckpt
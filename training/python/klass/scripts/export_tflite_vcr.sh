CUDA_VISIBLE_DEVICES=-1 python tools/export_tflite.py \
    --no-per_channel \
    --out_path klass_vcr.tflite \
    --config configs/vcr.yml \
    --calibration_dataset datasets/vcr/calibration/dataset.txt \
    --weights saved_models/vcr.pth

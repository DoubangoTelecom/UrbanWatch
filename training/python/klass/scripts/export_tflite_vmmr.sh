CUDA_VISIBLE_DEVICES=-1 python tools/export_tflite.py \
    --no-per_channel \
    --out_path klass_vmmr.tflite \
    --config configs/vmmr.yml \
    --calibration_dataset datasets/vmmr/calibration/dataset.txt \
    --weights saved_models/vmmr.pth

CUDA_VISIBLE_DEVICES=-1 python tools/export_tflite.py \
    --no-per_channel \
    --out_path klass_vbsr.tflite \
    --config configs/vbsr.yml \
    --calibration_dataset datasets/vbsr/calibration/dataset.txt \
    --weights saved_models/vbsr.pth

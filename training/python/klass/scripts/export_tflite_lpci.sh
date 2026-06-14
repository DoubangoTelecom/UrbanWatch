CUDA_VISIBLE_DEVICES=-1 python tools/export_tflite.py \
    --no-per_channel \
    --out_path klass_lpci.tflite \
    --config configs/lpci.yml \
    --calibration_dataset datasets/lpci/calibration/dataset.txt \
    --weights saved_models/lpci.pth

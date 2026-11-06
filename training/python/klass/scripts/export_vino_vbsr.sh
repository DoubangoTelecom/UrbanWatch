CUDA_VISIBLE_DEVICES=-1 python tools/export_vino.py \
    --out_path vino_models/klass_vbsr.vino/model.xml \
    --cfg_path configs/vbsr.yml \
    --calibration_dataset datasets/vbsr/calibration \
    --weights saved_models/vbsr.pth
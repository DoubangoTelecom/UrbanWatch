CUDA_VISIBLE_DEVICES=-1 python tools/export_vino.py \
    --out_path vino_models/klass_vmmr.vino/model.xml \
    --cfg_path configs/vmmr.yml \
    --calibration_dataset datasets/vmmr/calibration \
    --weights saved_models/vmmr.pth
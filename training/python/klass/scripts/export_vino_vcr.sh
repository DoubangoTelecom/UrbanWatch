CUDA_VISIBLE_DEVICES=-1 python tools/export_vino.py \
    --out_path vino_models/klass_vcr.vino/model.xml \
    --cfg_path configs/vcr.yml \
    --calibration_dataset datasets/vcr/calibration \
    --weights saved_models/vcr.pth
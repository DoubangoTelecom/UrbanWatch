CUDA_VISIBLE_DEVICES=-1 python tools/export_vino.py \
    --out_path vino_models/klass_lpci.vino/model.xml \
    --cfg_path configs/lpci.yml \
    --calibration_dataset datasets/lpci/calibration \
    --weights saved_models/lpci.pth
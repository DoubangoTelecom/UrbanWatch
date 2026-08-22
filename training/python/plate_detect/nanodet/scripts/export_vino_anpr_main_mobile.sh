CUDA_VISIBLE_DEVICES=-1 python tools/export_vino.py \
    --out_path vino_models/anpr_detect_main_mobile.vino/model.xml \
    --cfg_path config/anpr_mobile.yml \
    --model_path workspace/anpr_mobile/model_best/model_best.ckpt

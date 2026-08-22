CUDA_VISIBLE_DEVICES=-1 python tools/export_vino.py \
    --out_path vino_models/anpr_detect_pysearch.vino/model.xml \
    --cfg_path config/anpr_pysearch.yml \
    --model_path workspace/anpr_pysearch/model_best/model_best.ckpt

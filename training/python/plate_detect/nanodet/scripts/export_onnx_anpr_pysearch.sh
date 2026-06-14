CUDA_VISIBLE_DEVICES=-1 python tools/export_onnx.py \
    --out_path anpr_detect_pysearch.onnx \
    --cfg_path config/anpr_pysearch.yml \
    --model_path workspace/anpr_pysearch/model_best/model_best.ckpt

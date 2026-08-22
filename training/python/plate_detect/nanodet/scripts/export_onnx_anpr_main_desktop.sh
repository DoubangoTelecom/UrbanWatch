CUDA_VISIBLE_DEVICES=-1 python tools/export_onnx.py \
    --out_path anpr_detect_main_desktop.onnx \
    --cfg_path config/anpr_desktop.yml \
    --model_path workspace/anpr_desktop/model_best/model_best.ckpt

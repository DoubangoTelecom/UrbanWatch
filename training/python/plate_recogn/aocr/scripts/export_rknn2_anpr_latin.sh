CUDA_VISIBLE_DEVICES=-1 python aocr/tools/export_rknn2.py \
    --platforms rk3588,rk3588s,rv1103b,rv1106b,rk3566,rk3568,rv1103,rv1106,rk3576,rk3562,rk3576,rv1126b \
    --target anpr_latin \
    --per_channel True \
    --quantized_algorithm normal \
    --out_folder rknn2_models \
    --calibration_dataset datasets/latin/calibration/dataset.txt \
    --cfg_path configs/config_anpr_latin.yml \
    --model_path saved_models/anpr_latin/best_norm_ED.pth

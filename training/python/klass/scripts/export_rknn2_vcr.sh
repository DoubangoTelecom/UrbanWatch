CUDA_VISIBLE_DEVICES=-1 python tools/export_rknn2.py \
    --platforms rk3588,rk3588s,rv1103b,rv1106b,rk3566,rk3568,rv1103,rv1106,rk3576,rk3562,rk3576,rv1126b \
    --target vcr \
    --per_channel True \
    --quantized_algorithm normal \
    --out_folder rknn2_models \
    --calibration_dataset datasets/vcr/calibration/dataset.txt \
    --cfg_path configs/vcr.yml \
    --model_path saved_models/vcr.pth

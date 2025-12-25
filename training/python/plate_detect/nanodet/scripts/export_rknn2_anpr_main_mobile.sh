CUDA_VISIBLE_DEVICES=-1 python tools/export_rknn2.py \
    --platforms rk3588,rk3588s,rv1103b,rv1106b,rk3566,rk3568,rv1103,rv1106,rk3576,rk3562,rk3576,rv1126b \
    --target main_mobile \
    --output_dynamic_shape False \
    --per_channel True \
    --out_folder rknn2_models \
    --calibration_dataset datasets/anpr_main/calibration/dataset.txt \
    --cfg_path config/anpr_mobile.yml \
    --model_path /home/projects/urban-watch/plate_detect/nanodet/workspace/anpr_mobile/model_best/model_best.ckpt

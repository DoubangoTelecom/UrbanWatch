import torch, argparse, os, onnx, onnxsim, shutil
from nanodet.model.arch import build_model
from nanodet.util import Logger, cfg, load_config, load_model_weight

BATCH_SIZE = 2

def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Convert .pth or .ckpt model to rknn2 plus quantization.",
    )
    parser.add_argument("--cfg_path", type=str, required=True, help="Path to .yml config file.")
    parser.add_argument("--target", type=str, required=True, help="Must be 'main_mobile' or 'pysearch'")
    parser.add_argument(
        "--model_path", type=str, required=True, help="Path to .ckpt model."
    )
    parser.add_argument(
        "--calibration_dataset", type=str, required=True, help="TXT file listing images to use for calibration. Built using scripts/calib_*.py"
    )
    parser.add_argument(
        "--out_folder", type=str, default="hailo_models", help="Onnx model output path."
    )
    parser.add_argument(
        "--platforms", type=str, default='hailo8r,hailo15h,hailo15m,hailo10h,hailo8,hailo8l', help="Comma separated list of platforms"
    )
    return parser.parse_args() 
           
def main(cfg, args):
    assert args.target in ['main_mobile', 'pysearch'], '--target value must be "main_mobile" or "pysearch"'
    logger = Logger(-1, cfg.save_dir, False)
    model = build_model(cfg.model).eval()
    checkpoint = torch.load(args.model_path, map_location=lambda storage, loc: storage, weights_only=False)
    load_model_weight(model, checkpoint, logger)
    if cfg.model.arch.backbone.name == "RepVGG":
        deploy_config = cfg.model
        deploy_config.arch.backbone.update({"deploy": True})
        deploy_model = build_model(deploy_config)
        from nanodet.model.backbone.repvgg import repvgg_det_model_convert

        model = repvgg_det_model_convert(model, deploy_model)
        
    # Dummy input
    sample_args = (torch.randn(BATCH_SIZE, 3, *cfg.data.val.input_size), )
    
    # Create output folder
    if not os.path.exists(args.out_folder):
        os.makedirs(args.out_folder)
    os.chmod(args.out_folder, 0o777)
    
    # Using quantized TFlite models as input doesn't work with RKNN2. That's why we use ONNX.
    onnx_model_path = os.path.join(args.out_folder, 'temp.onnx')
    torch.onnx.export(
        model,
        sample_args,
        onnx_model_path,
        verbose=False,
        opset_version=11, # OpeSet 11 cause warnings
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'} } if BATCH_SIZE > 1 else None,
    )
    onnx_model_simplified, flag = onnxsim.simplify(onnx_model_path)
    assert flag, 'Failed to simplify the ONNX model'
    onnx.save(onnx_model_simplified, onnx_model_path)
    
    model_script = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../scripts/', f'hailo_anpr_{args.target}.alls')
    platforms = args.platforms.split(',')
    assert len(platforms) > 0, "Empty list of platforms"
    har_file = os.path.join(args.out_folder, 'temp.har')
    for i, platform in enumerate(platforms):
        print('[{:d}/{:2d}] Processing platform {}'.format(i, len(platforms), platform))
        hailo_folder = os.path.join(args.out_folder, platform)
        model_folder = os.path.join(hailo_folder, f'anpr_detect_{args.target}')
        os.makedirs(model_folder)
        assert os.system(f'yes n | hailo parser onnx {onnx_model_path} --har-path {har_file} --hw-arch {platform} --input-format input=NCHW') == 0, 'Parser failed'
        assert os.system(f'hailo optimize {har_file} --output-har-path {har_file} --hw-arch {platform} --calib-set-path {args.calibration_dataset} --model-script {model_script}') == 0, 'Optimizer failed'
        assert os.system(f'hailo compiler {har_file} --hw-arch {platform} --output-dir {model_folder}') == 0, 'Compiler failed'
        shutil.move(os.path.join(model_folder, 'temp.hef'), os.path.join(hailo_folder, f'anpr_detect_{args.target}.{platform}'))
        shutil.rmtree(model_folder)
        
    # CleanUp
    os.remove(onnx_model_path)
    os.remove(har_file)
           
if __name__ == "__main__":
    print("Execute this file using [hailo] conda env on the RTX4060 machine")
    args = parse_args()
    cfg_path = args.cfg_path
    load_config(cfg, cfg_path)
    main(cfg, args)
    print('!!!Done!!!')
    
    
    
    

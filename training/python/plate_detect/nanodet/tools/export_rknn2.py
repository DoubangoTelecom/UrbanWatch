import torch, argparse, os, onnx, onnxsim
from rknn.api import RKNN
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
        "--calibration_dataset", type=str, required=True, help="TXT file listing images to use for calibration. Built using 'tools/calib_build.py'"
    )
    parser.add_argument(
        "--out_folder", type=str, default="rknn_models", help="Onnx model output path."
    )
    parser.add_argument(
        "--platforms", type=str, default='rk3588s,rv1103b,rv1106b,rk3566,rk3568,rv1103,rv1106,rk3588,rk3576,rk3562,rk3576,rv1126b', help="Comma separated list of platforms"
    ) # FIXME(dmi): 'rk2118' doesn't work
    parser.add_argument(
        "--optimization_level", type=int, default=3, help="Optimization level [0-3]."
    )
    parser.add_argument("--per_channel", required=False, default=True, help="Whether to perform per channel quantization.")
    parser.add_argument("--quantized_algorithm", required=False, default='normal', help="Quantized algorithm.") # FIXME(dmi): 'mmse' hangs
    return parser.parse_args() 
           
def main(cfg, args):
    assert args.target in ['main_mobile', 'pysearch'], '--target value must be "main_mobile" or "pysearch"'
    logger = Logger(-1, cfg.save_dir, False)
    model = build_model(cfg.model)
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
    onnx_model_path = os.path.join(args.out_folder, '____model___.onnx')
    torch.onnx.export(
        model.eval(),
        sample_args,
        onnx_model_path,
        verbose=False,
        keep_initializers_as_inputs=True,
        opset_version=18, # OpeSet 11 cause warnings
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'} },
    )
    onnx_model_simplified, flag = onnxsim.simplify(onnx_model_path)
    assert flag, 'Failed to simplify the ONNX model'
    onnx.save(onnx_model_simplified, onnx_model_path)
    
    platforms = args.platforms.split(',')
    assert len(platforms) > 0, "Empty list of platforms"
    for i, platform in enumerate(platforms):
        print('[{:d}/{:2d}] Processing platform {}'.format(i, len(platforms), platform))
        rknn = RKNN(verbose=False)
        rknn.config(
            target_platform=platform,
            mean_values=cfg.data.val.pipeline.normalize[0], 
            std_values=cfg.data.val.pipeline.normalize[1],
            quant_img_RGB2BGR=False, 
            optimization_level=args.optimization_level,
            quantized_method={'True': 'channel', 'False': 'layer'}[args.per_channel],
            quantized_algorithm=args.quantized_algorithm
        )
        print('Loading ONNX model...')
        ret = rknn.load_onnx(model=onnx_model_path)
        assert ret == 0, 'Failed to load ONNX model from {} with error code {} for platform {}'.format(onnx_model_path, ret, platform)
        print('Building...')
        ret = rknn.build(do_quantization=True, dataset=args.calibration_dataset)
        assert ret == 0, 'Failed to build ONNX model at {} with error code {} for platform {}'.format(onnx_model_path, ret, platform)
        print('Exporting....')
        rknn_folder = os.path.join(args.out_folder, platform)
        if not os.path.exists(rknn_folder):
            os.makedirs(rknn_folder)
        ret = rknn.export_rknn(os.path.join(rknn_folder, 'anpr_detect_{}.rknn.{}'.format(args.target, platform)))
        assert ret == 0, 'Failed to export ONNX model at {} with error code {} for platform {}'.format(onnx_model_path, ret, platform)
        
    # CleanUp
    os.remove(onnx_model_path)
           
if __name__ == "__main__":
    print("Execute this file using [rknn2] conda env on the RTX4060 machine")
    args = parse_args()
    cfg_path = args.cfg_path
    load_config(cfg, cfg_path)
    main(cfg, args)
    print('!!!Done!!!')
    
    
    
    
# Deprecated: "ai_edge_torch" doesn't support batch size > 1
import torch, argparse, os, numpy as np
from PIL import Image
from nanodet.model.arch import build_model
from nanodet.util import Logger, cfg, load_config, load_model_weight

import ai_edge_torch, tensorflow as tf
from ai_edge_torch.quantize import pt2e_quantizer
from ai_edge_torch.quantize import quant_config

from torch.ao.quantization import quantize_pt2e

from export_utils import get_image_list, load_image_then_preprocess

def tflite_quant_representative_dataset(paths, cfg):
    for i, path in enumerate(paths):
        print('[Round #2][{:3d}/{:3d}] Calibration using {}'.format(i, len(paths), path))
        image = load_image_then_preprocess(path, cfg)
        yield [image]
           
def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Convert .pth or .ckpt model to tflite plus quantization.",
    )
    parser.add_argument("--cfg_path", type=str, help="Path to .yml config file.")
    parser.add_argument(
        "--model_path", type=str, default=None, help="Path to .ckpt model."
    )
    parser.add_argument(
        "--out_path", type=str, default="nanodet.tflite", help="Onnx model output path."
    )
    parser.add_argument(
        "--input_shape", type=str, default=None, help="Model intput shape."
    )
    parser.add_argument("--output_dynamic_shape", required=False, default=False, help="Whether to enable dynamic shape for the output.")
    parser.add_argument("--per_channel", required=False, default=False, help="Whether to perform per channel quantization.")
    return parser.parse_args() 
           
def main(cfg, model_path, out_path, input_shape, per_channel=False, output_dynamic_shape=False):
    logger = Logger(-1, cfg.save_dir, False)
    model = build_model(cfg.model)
    checkpoint = torch.load(model_path, map_location=lambda storage, loc: storage, weights_only=False)
    load_model_weight(model, checkpoint, logger)
    if cfg.model.arch.backbone.name == "RepVGG":
        deploy_config = cfg.model
        deploy_config.arch.backbone.update({"deploy": True})
        deploy_model = build_model(deploy_config)
        from nanodet.model.backbone.repvgg import repvgg_det_model_convert

        model = repvgg_det_model_convert(model, deploy_model)
    
    paths = get_image_list(cfg.data.val.img_path)
    assert len(paths) > 0, "List of images is empty"
    print('Number of images for calibration:', len(paths))
    
    # Dynamic shape doesn't work. The batch size (B)
    # is lost somewhere and the output will always
    # has B=1
    assert not output_dynamic_shape, 'Dynamic shape not supported'
    batch = torch.export.Dim('batch', min=1, max=50)
    
    # Per-channel quantization generate larger models which are
    # slower on NPU.
    if per_channel:
        print('Per channel quatization is very slow on NPU') # really???
        
    # Dummy input
    sample_args = (torch.randn(1, 3, *input_shape), )
    
    quantizer = pt2e_quantizer.PT2EQuantizer().set_global(
        pt2e_quantizer.get_symmetric_quantization_config(is_per_channel=per_channel, is_dynamic=False, is_qat=False)
    )
    
    model = torch.export.export(model.eval(), sample_args).module()
    model = quantize_pt2e.prepare_pt2e(model, quantizer)
    
    # Initialize internal states (!!!required!!!)
    model(*sample_args)
    
    # Here is some crazy shit: I have poor results if I don't
    # perform first round quantization here: doesn't make sense,
    # I'm already providing a representative dataset via "tfl_converter_flags"
    for i, path in enumerate(paths):
        print('[Round #1][{:3d}/{:3d}] Calibration using {}'.format(i, len(paths), path))
        image = load_image_then_preprocess(path, cfg)
        model(torch.from_numpy(image))
    
    torch.ao.quantization.move_exported_model_to_eval(model)
    model = quantize_pt2e.convert_pt2e(model, fold_quantize=False)
    
    tfl_converter_flags = {
            'optimizations': [tf.lite.Optimize.DEFAULT], 
            'representative_dataset': lambda: tflite_quant_representative_dataset(paths, cfg),
            'constant_folding': True,
            'target_spec': {
                'supported_ops': [tf.lite.OpsSet.TFLITE_BUILTINS_INT8],
                'supported_types': [tf.int8],
            },
            'inference_input_type': tf.float32,
            'inference_output_type': tf.float32,
            "_experimental_disable_per_channel": not per_channel,
            "_experimental_new_quantizer": True,
            # Enable more aggressive quantization analysis
            "_experimental_calibrate_quantization": True,
    }
    
    with_quantizer = ai_edge_torch.convert(
        model,
        sample_args,
        quant_config=quant_config.QuantConfig(pt2e_quantizer=quantizer),
        _ai_edge_converter_flags=tfl_converter_flags,
        dynamic_shapes={'data': {0: batch}} if output_dynamic_shape else None
    )
    
    with_quantizer.export(out_path)
           
if __name__ == "__main__":
    print("Execute this file using [tflite] conda env on the RTX4060 machine")
    args = parse_args()
    cfg_path = args.cfg_path
    model_path = args.model_path
    out_path = args.out_path
    input_shape = args.input_shape
    load_config(cfg, cfg_path)
    if input_shape is None:
        input_shape = cfg.data.val.input_size
    else:
        input_shape = tuple(map(int, input_shape.split(",")))
        assert len(input_shape) == 2
    if model_path is None:
        model_path = os.path.join(cfg.save_dir, "model_best/model_best.ckpt")
    main(cfg, model_path, out_path, input_shape, args.per_channel=='True', args.output_dynamic_shape=='True')
    print("Model saved to:", out_path)
    
    
    
    
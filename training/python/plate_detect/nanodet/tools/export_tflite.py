# Debix A: .:$LD_LIBRARY_PATH VIV_VX_DEBUG_LEVEL=0 VIV_VX_PROFILE=0 ./linux_aarch64_benchmark_model --graph="inference_models/anpr_detect_main_mobile.tflite" --num_threads=4 --external_delegate_path="/usr/lib/libvx_delegate.so" --input_layer_shape=10,3,320,320 --input_layer=input
import torch, argparse, os, onnxsim, onnx, shutil
import tensorflow as tf
from nanodet.model.arch import build_model
from nanodet.util import Logger, cfg, load_config, load_model_weight

from export_utils import get_image_list, load_image_then_preprocess

BATCH_SIZE = 2 # set to >1 to force batching
ONNX_FILE = '____model___.onnx' # ONNX temp file
TF_FOLDER = 'tf_saved_model'

def tflite_quant_representative_dataset(base_folder, fnames, cfg):
    for i, fname in enumerate(fnames):
        print('[{:3d}/{:3d}] Calibration using {}'.format(i, len(fnames), fname))
        image = load_image_then_preprocess(os.path.join(base_folder, fname), cfg, channel_first=True)
        yield [image]
           
def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Convert .pth or .ckpt model to tflite plus quantization.",
    )
    parser.add_argument("--cfg_path", type=str, required=True, help="Path to .yml config file.")
    parser.add_argument(
        "--model_path", type=str, required=True, default=None, help="Path to .ckpt model."
    )
    parser.add_argument(
        "--calibration_dataset", type=str, required=True, help="TXT file listing images to use for calibration. Built using 'tools/calib_build.py'"
    )
    parser.add_argument(
        "--out_path", type=str, default="nanodet.tflite", help="TFLite model output path."
    )
    parser.add_argument(
        "--input_shape", type=str, default=None, help="Model intput shape."
    )
    parser.add_argument("--per_channel", required=False, default=False, help="Whether to perform per channel quantization.")
    return parser.parse_args() 
           
def main(cfg, model_path, out_path, calibration_dataset, input_shape, per_channel=False):
    logger = Logger(-1, cfg.save_dir, False)
    model = build_model(cfg.model).eval()
    checkpoint = torch.load(model_path, map_location=lambda storage, loc: storage, weights_only=False)
    load_model_weight(model, checkpoint, logger)
    if cfg.model.arch.backbone.name == "RepVGG":
        deploy_config = cfg.model
        deploy_config.arch.backbone.update({"deploy": True})
        deploy_model = build_model(deploy_config)
        from nanodet.model.backbone.repvgg import repvgg_det_model_convert

        model = repvgg_det_model_convert(model, deploy_model)
    
    # Read images
    with open(calibration_dataset) as f:
        fnames = f.read().splitlines()
    assert len(fnames) > 0, "List of images is empty"
    print('Number of images for calibration:', len(fnames))
    
    # Per-channel quantization generate larger models which are
    # slower on NPU.
    if per_channel:
        print('Per channel quatization is very slow on NPU') # really???
        
    # Dummy input
    sample_args = (torch.randn(BATCH_SIZE, 3, *input_shape), )
    
    # Convert to ONNX
    print('Exporting to ONNX [{}]...'.format(ONNX_FILE))
    if os.path.exists(ONNX_FILE):
        os.remove(ONNX_FILE)
    torch.onnx.export(
        model,
        sample_args,
        ONNX_FILE,
        verbose=False,
        opset_version=11,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'} },
    )
    onnx_model_simplified, flag = onnxsim.simplify(ONNX_FILE)
    assert flag, 'Failed to simplify the ONNX model'
    onnx.save(onnx_model_simplified, ONNX_FILE)
    
    # Convert to Tensorflow. 
    # '-k' option to keep the input format as NCHW to ease C++ code.
    print('Exporting to Tensorflow [{}]...'.format(TF_FOLDER))
    if os.path.exists(TF_FOLDER):
        shutil.rmtree(TF_FOLDER)
    assert os.system('onnx2tf -i {} -o {} -k {} -otfv1pb'.format(ONNX_FILE, TF_FOLDER, 'input')) == 0, 'Failed to convert to Tensorflow'
    
    # Quantization https://www.tensorflow.org/lite/performance/post_training_integer_quant
    
    converter = tf.lite.TFLiteConverter.from_saved_model(TF_FOLDER)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = lambda: tflite_quant_representative_dataset(os.path.dirname(calibration_dataset), fnames, cfg)
    converter.constant_folding = True
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.target_spec.supported_types = [tf.int8]
    converter.inference_input_type = tf.float32
    converter.inference_output_type = tf.float32
    converter._experimental_disable_per_channel = not per_channel
    converter._experimental_new_quantizer = True
    # Enable more aggressive quantization analysis
    converter._experimental_calibrate_quantization = True
    
    tflite_model_quant = converter.convert()
    with open(out_path, 'wb') as f:
        f.write(tflite_model_quant)
    
    # Delete tmp files
    os.remove(ONNX_FILE)
    shutil.rmtree(TF_FOLDER)
    
           
if __name__ == "__main__":
    print("Execute this file using [onnx2tf] conda env on the RTX4060 machine")
    args = parse_args()
    cfg_path = args.cfg_path
    model_path = args.model_path
    input_shape = args.input_shape
    load_config(cfg, cfg_path)
    if input_shape is None:
        input_shape = cfg.data.val.input_size
    else:
        input_shape = tuple(map(int, input_shape.split(",")))
        assert len(input_shape) == 2
    if model_path is None:
        model_path = os.path.join(cfg.save_dir, "model_best/model_best.ckpt")
    main(cfg, model_path, args.out_path, args.calibration_dataset, input_shape, args.per_channel=='True')
    print("Model saved to:", args.out_path)
    
    
    
    
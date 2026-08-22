# Debix A: .:$LD_LIBRARY_PATH VIV_VX_DEBUG_LEVEL=0 VIV_VX_PROFILE=0 ./linux_aarch64_benchmark_model --graph="inference_models/klass_vcr.tflite" --num_threads=4 --external_delegate_path="/usr/lib/libvx_delegate.so" --input_layer_shape=10,3,128,128 --input_layer=input
import torch, argparse, os, onnxsim, onnx, shutil, numpy as np
import tensorflow as tf
from klass.model import KlassModel
from klass.config import Config

from export_utils import load_image_then_preprocess

BATCH_SIZE = 2 # set to >1 to force batching
ONNX_FILE = '____model___.onnx' # ONNX temp file
TF_FOLDER = 'tf_saved_model'

def tflite_quant_representative_dataset(base_folder, fnames, cfg):
    for i, fname in enumerate(fnames):
        print('[{:3d}/{:3d}] Calibration using {}'.format(i, len(fnames), fname))
        image = load_image_then_preprocess(os.path.join(base_folder, fname), cfg, channel_first=True)
        yield [image]
         
def export(cfg, opt):
    # Build Model
    model = KlassModel(cfg, training=False).eval()
    model.load_state_dict(torch.load(opt.weights, map_location=lambda storage, loc: storage, weights_only=True))
    
    # Read images
    with open(opt.calibration_dataset) as f:
        fnames = f.read().splitlines()
    assert len(fnames) > 0, "List of images is empty"
    print('Number of images for calibration:', len(fnames))
    
    # Per-channel quantization generate larger models which are
    # slower on NPU.
    if opt.per_channel:
        print('Per channel quatization is very slow on NPU') # really???
        
    # Dummy input
    sample_args = (torch.randn(BATCH_SIZE, 1 if cfg.model.chroma == 'gray' else 3, cfg.model.imgH, cfg.model.imgW), )
    
    # Convert to ONNX
    print('Exporting to ONNX [{}]...'.format(ONNX_FILE))
    if os.path.exists(ONNX_FILE):
        os.remove(ONNX_FILE)    
    torch.onnx.export(
        model,
        *sample_args,
        ONNX_FILE,
        verbose=True,
        opset_version=11, # requires old torch (e.g. 1.13.1), newest one (e.g. 2.9.0) will force us to use opset-18 which is not supported by jetpack 4.x
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={'input': {0: 'batch_size'},
                    'output': {0: 'batch_size'}},
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
    converter.representative_dataset = lambda: tflite_quant_representative_dataset(os.path.dirname(opt.calibration_dataset), fnames, cfg)
    converter.constant_folding = True
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.target_spec.supported_types = [tf.int8]
    converter.inference_input_type = tf.float32
    converter.inference_output_type = tf.float32
    converter._experimental_disable_per_channel = not opt.per_channel
    converter._experimental_new_quantizer = True
    # Enable more aggressive quantization analysis
    converter._experimental_calibrate_quantization = True
    
    tflite_model_quant = converter.convert()
    with open(opt.out_path, 'wb') as f:
        f.write(tflite_model_quant)
    
    # Delete tmp files
    os.remove(ONNX_FILE)
    shutil.rmtree(TF_FOLDER)
    
           
if __name__ == "__main__":
    print('Use [onnx2tf] conda env on RTX4060')
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True, help="path to config file")
    parser.add_argument('--weights', required=True, help="path to models's weigths")
    parser.add_argument(
        "--calibration_dataset", type=str, required=True, help="TXT file listing images to use for calibration. Built using 'tools/calib_build.py'"
    )  
    parser.add_argument("--per_channel", action=argparse.BooleanOptionalAction, required=False, default=False, help="Whether to perform per channel quantization.")
    parser.add_argument("--out_path", type=str, default="klass_xxx.tflite", required=True, help="TFLite model output path.")

    opt = parser.parse_args()
    
    cfg = Config.parse(opt.config)

    export(cfg, opt)
    
    print('!!! DONE !!!')
    
    
    
    
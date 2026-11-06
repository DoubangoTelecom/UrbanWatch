import torch, argparse, onnxsim, onnx, os, gzip, numpy as np

from aocr.model import AOCR
from aocr.config import Config

from modelopt.onnx.quantization import quantize

BATCH_SIZE = 2 # must be > 1, otherwise exported model not 'batchy'

def export(cfg, opt):
    # Build Model
    model = AOCR(cfg, training=False).eval()
    model.load_state_dict(torch.load(opt.weights, map_location=lambda storage, loc: storage, weights_only=True))
    
    sample_args = (torch.randn(BATCH_SIZE, 1 if cfg.model.grayscale else 3, cfg.model.imgH, cfg.model.imgW), )
    modes = ['all']#['-stn', 'stn'] if cfg.model.stn_type != "none" else ['-stn']
    for mode in modes:
        print('Exporting for mode [{}]...'.format(mode))
        model._set_export_mode(mode)
        onnx_file_path = 'aocr_{}-stn.temp.onnx'.format(cfg.model.name) if mode == 'stn' else \
            'aocr_{}.temp.onnx'.format(cfg.model.name)
        
        torch.onnx.export(
            model.eval(),
            *sample_args,
            onnx_file_path,
            verbose=False,
            keep_initializers_as_inputs=True,
            opset_version=18 if ('all' in modes) else 11, # requires old torch (e.g. 1.13.1), newest one (e.g. 2.9.0) will force us to use opset-18 which is not supported by jetpack 4.x
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={'input': {0: 'batch_size'},
                        'output': {0: 'batch_size'}},
        )

        print("start simplifying onnx...")
        input_data = {"data": sample_args[0].detach().cpu().numpy()}
        model_sim, flag = onnxsim.simplify(onnx_file_path, input_data=input_data, check_n=0)
        if flag:
            onnx.save(model_sim, onnx_file_path)
            print("simplify onnx successfully")
        else:
            print("simplify onnx failed")

        # Quantize
        with gzip.open(opt.calibration_dataset, 'r') as f:
            quantize(
                onnx_path=onnx_file_path,
                quantize_mode="int8",       # fp8, int8, int4 etc.
                calibration_data=np.load(f),
                calibration_method="entropy",   # max, entropy, awq_clip, rtn_dq etc.
                output_path=onnx_file_path.replace('.temp.onnx', '.qnnx'),
                autotune=True,
                simplify=True
            )

        # remove onnx temp files
        os.remove(onnx_file_path)
        os.remove('{}.data'.format(onnx_file_path))


if __name__ == '__main__':
    print('Use [modelopt] conda env on RTX4060')
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True, help="path to config file")
    parser.add_argument('--weights', required=True, help="path to models's weigths")
    parser.add_argument("--calibration_dataset", type=str, required=True, help="*.npy.gz")

    opt = parser.parse_args()
    
    cfg = Config.parse(opt.config)

    export(cfg, opt)

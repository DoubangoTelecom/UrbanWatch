import torch, argparse, onnxsim, onnx, os

from klass.model import KlassModel
from klass.config import Config

BATCH_SIZE = 2 # must be > 1, otherwise exported model not 'batchy'

def export(cfg, opt):
    # Build Model
    model = KlassModel(cfg, training=False).eval()
    model.load_state_dict(torch.load(opt.weights, map_location=lambda storage, loc: storage, weights_only=True))
    
    sample_args = (torch.randn(BATCH_SIZE, 1 if cfg.model.chroma == 'gray' else 3, cfg.model.imgH, cfg.model.imgW), )
    onnx_file_path = f'klass_{cfg.model.name}.onnx'

    torch.onnx.export(
        model.eval(),
        *sample_args,
        onnx_file_path,
        verbose=False,
        keep_initializers_as_inputs=True,
        opset_version=11, # requires old torch (e.g. 1.13.1), newest one (e.g. 2.9.0) will force us to use opset-18 which is not supported by jetpack 4.x
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={'input': {0: 'batch_size'},
                    'output': {0: 'batch_size'}},
    )

    print("start simplifying onnx...")
    input_data = {"data": sample_args[0].detach().cpu().numpy()}
    model_sim, flag = onnxsim.simplify(onnx_file_path, input_data=input_data)
    if flag:
        onnx.save(model_sim, onnx_file_path)
        print("simplify onnx successfully")
    else:
        print("simplify onnx failed")


if __name__ == '__main__':
    print('Use [onnx2tf] conda env on RTX4060')
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True, help="path to config file")
    parser.add_argument('--weights', required=True, help="path to models's weigths")

    opt = parser.parse_args()
    
    cfg = Config.parse(opt.config)

    export(cfg, opt)

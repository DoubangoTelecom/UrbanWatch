import torch, argparse, os, numpy as np, nncf, openvino as ov, onnxsim
from io import BytesIO
from aocr.model import AOCR
from aocr.config import Config
from aocr.dataset import RawDataset, AlignCollate

BATCH_SIZE = 2 # set to >1 to make it batchy
ONNX_FILE = '____model___.onnx'
          
def main(cfg, opt):
    # Build Model
    model = AOCR(cfg, training=False).eval()
    model.load_state_dict(torch.load(opt.weights, map_location=lambda storage, loc: storage, weights_only=True))
        
    sample_args = (torch.randn(BATCH_SIZE, 1 if cfg.model.grayscale else 3, cfg.model.imgH, cfg.model.imgW), )
    
    # We have issues quantizing the pytorch model directly to OpenVINO.
    # That's why we convert it to OpenVINO without quantization, then
    # we apply quantization.
    if False:
        # Got AttributeError: module 'openvino' has no attribute 'Node'. Did you mean: 'Model'?
        ov_model = ov.convert_model(model.eval(), example_input=sample_args)
    else:
        if os.path.exists(ONNX_FILE):
            os.remove(ONNX_FILE)
        
        torch.onnx.export(
            model.eval(),
            sample_args,
            ONNX_FILE,
            verbose=False,
            keep_initializers_as_inputs=True,
            opset_version=18, # OpeSet 11 cause warnings
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'} },
        )
        ov_model, flag = onnxsim.simplify(ONNX_FILE)
        os.remove(ONNX_FILE)
        os.remove('{}.data'.format(ONNX_FILE))
        assert flag, 'Failed to simplify the ONNX model'
        
        input_name = ov_model.graph.input[0].name
        def transform_fn(data_item):
            tensor, path = data_item
            return { input_name: tensor.numpy() }
    
    # Dataset
    AlignCollate_demo = AlignCollate(cfg)
    demo_data = RawDataset(root=os.path.dirname(opt.calibration_dataset), opt=cfg)  # use RawDataset
    train_dataloader = torch.utils.data.DataLoader(
        demo_data, batch_size=1,
        shuffle=False,
        num_workers=0,
        collate_fn=AlignCollate_demo, pin_memory=True)
    
    # Quantization
    calibration_dataset = nncf.Dataset(train_dataloader, transform_fn)
    quantized_model = nncf.quantize(
        model=ov_model, 
        calibration_dataset=calibration_dataset,
        preset=nncf.QuantizationPreset.MIXED
    )
    ov_quantized_model = ov.convert_model(BytesIO(quantized_model.SerializeToString()), example_input=sample_args)
    
    # Change Input/Output type
    ppp = ov.preprocess.PrePostProcessor(ov_quantized_model)
    ppp.input().tensor() \
        .set_element_type(ov.Type.f32) \
        .set_layout(ov.Layout('NCHW'))
    ppp.output().tensor() \
        .set_element_type(ov.Type.f32) \
        .set_layout(ov.Layout('NCHW'))

    # save the model
    ov.save_model(ov_quantized_model, opt.out_path)
    
    # Change permission to allow delete
    os.chmod(opt.out_path, 0o777)
           
if __name__ == "__main__":
    print("Execute this file using [vino] conda env on the RTX4060 machine")
    
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Convert .pth or .ckpt model to tflite plus quantization.",
    )
    parser.add_argument("--cfg_path", required=True, type=str, help="Path to .yml config file.")
    parser.add_argument('--weights', required=True, help="path to models's weigths")
    parser.add_argument(
        "--out_path", type=str, required=True, default="aocr.xml", help="Onnx model output path."
    )
    parser.add_argument(
        "--calibration_dataset", type=str, required=True, help="TXT file listing images to use for calibration. Built using 'tools/calib_build.py'"
    )

    opt = parser.parse_args()
    
    cfg = Config.parse(opt.cfg_path)

    main(cfg, opt)
    
    
    
    
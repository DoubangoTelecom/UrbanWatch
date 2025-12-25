import torch, argparse, os, numpy as np, nncf, openvino as ov, onnxsim
from io import BytesIO
from nanodet.model.arch import build_model
from nanodet.data.dataset import build_dataset
from nanodet.util import Logger, cfg, load_config, load_model_weight
from export_utils import get_image_list, preprocess_image

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
        "--out_path", type=str, default="nanodet.xml", help="Onnx model output path."
    )
    parser.add_argument(
        "--input_shape", type=str, default=None, help="Model intput shape."
    )
    parser.add_argument("--output_dynamic_shape", required=False, default=False, help="Whether to enable dynamic shape for the output.")
    return parser.parse_args() 
           
def main(cfg, model_path, out_path, input_shape, output_dynamic_shape=False):
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
    
    # Dynamic shape doesn't work. The batch size (B)
    # is lost somewhere and the output will always
    # has B=1
    assert not output_dynamic_shape, 'Dynamic shape not supported'
        
    # Dummy input
    sample_args = (torch.randn(1, 3, *input_shape), )
    
    # https://github.com/openvinotoolkit/nncf

    # Dummy input
    sample_args = (torch.randn(1, 3, cfg.data.val.input_size[0], cfg.data.val.input_size[1]), )
    
    # We have issues quantizing the pytorch model directly to OpenVINO.
    # That's why we convert it to OpenVINO without quantization, then
    # we apply quantization.
    if False:
        # Got AttributeError: module 'openvino' has no attribute 'Node'. Did you mean: 'Model'?
        ov_model = ov.convert_model(model.eval(), example_input=sample_args)
    else:
        torch.onnx.export(
            model.eval(),
            sample_args,
            '____model___.onnx',
            verbose=False,
            keep_initializers_as_inputs=True,
            opset_version=18, # OpeSet 11 cause warnings
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'} } if output_dynamic_shape else None,
        )
        ov_model, flag = onnxsim.simplify('____model___.onnx')
        os.remove('____model___.onnx')
        os.remove('____model___.onnx.data')
        assert flag, 'Failed to simplify the ONNX model'
        
        input_name = ov_model.graph.input[0].name
        def transform_fn(data_item):
            return { input_name: data_item['img'].numpy() }
    
    # Dataset
    train_dataset = build_dataset(cfg.data.train, "test")
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=False)
    
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
    ov.save_model(ov_quantized_model, out_path)
    
    # Change permission to allow delete
    os.chmod(out_path, 0o777)
           
if __name__ == "__main__":
    print("Execute this file using [vino] conda env on the RTX4060 machine")
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
    main(cfg, model_path, out_path, input_shape, args.output_dynamic_shape=='True')
    print("Model saved to:", out_path)
    
    
    
    
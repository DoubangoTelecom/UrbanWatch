import argparse, os, cv2, openvino as ov
from demo_utils import get_image_list, load_image
import torch
from nanodet.model.head import build_head
from nanodet.util import cfg, load_config

# https://docs.openvino.ai/2024/openvino-workflow/running-inference/inference-devices-and-modes/cpu-device.html
# https://docs.openvino.ai/2023.3/notebooks/002-openvino-api-with-output.html#openvino-ir-model

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="model config file path")
    parser.add_argument("--model", required=True, help="model file path")
    parser.add_argument("--inputs", required=True, default="./images", help="path to the input images")
    parser.add_argument("--outputs", required=False, default=None, help="path to the output images")
    parser.add_argument("--device_name", required=False, default="CPU", help="Target device name")
    parser.add_argument(
        "--show_result",
        action="store_true",
        help="whether to show the results to the screen (requires GUI API)",
    )
    parser.add_argument(
        "--save_result",
        action="store_true",
        help="whether to save the inference result of image",
    )
    args = parser.parse_args()
    return args

def main():  
    print('Run this demo on [vino] conda env on RTX4060 machine')
    # Parse arguments
    args = parse_args()
    # Load config
    load_config(cfg, args.config)
    
    # Create head for post-processing
    head = build_head(cfg['model']['arch']['head'])
    
    # Prepare VINO engine and print info
    core = ov.Core()
    model = core.read_model(args.model)
    input_layer = model.input(0)
    print(f"input precision: {input_layer.element_type}")
    print(f"input shape: {input_layer.shape}")
    
    # Compile for CPU
    compiled_model = core.compile_model(model=model, device_name=args.device_name)

    # Collect images
    if os.path.isdir(args.inputs):
        files = get_image_list(args.inputs)
    else:
        files = [args.inputs]
    files.sort()
    print('Number of images:', len(files))
    assert len(files) > 0, "List of input images is empty"
    
    # Loop through the images and do inference
    for i, path in enumerate(files):
        print('[{:3d}/{:3d}]Processing {}...'.format(i, len(files), path))
        # Pre-process and build meta
        meta = load_image(path, cfg)
        # Inference
        preds = compiled_model([meta['numpy_img']])
        assert len(preds) == 1, 'We expect a single output'
        # Post-processing
        results = head.post_process(torch.from_numpy(preds[0]), meta)
        # Visualize and save
        result_img = head.show_result(
            meta["raw_img"][0], results[0], cfg.class_names, score_thres=0.35, show=args.show_result
        )
        if args.save_result:
            assert(not args.outputs is None, 'Path to --outputs folder must be defined when --save_result is set')
            save_path =  os.path.join(args.outputs, os.path.basename(path))
            cv2.imwrite(save_path, result_img)       
            print('Result saved at {}'.format(save_path)) 
        if args.show_result:
            ch = cv2.waitKey(0)
            if ch == 27 or ch == ord("q") or ch == ord("Q"):
                break 

if __name__ == "__main__":
    main()
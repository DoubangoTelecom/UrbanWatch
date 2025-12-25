import argparse, os, onnxruntime as rt, cv2
from demo_utils import get_image_list, load_image
import torch
from nanodet.model.head import build_head
from nanodet.util import cfg, load_config

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="model config file path")
    parser.add_argument("--model", required=True, help="model file path")
    parser.add_argument("--inputs", required=True, default="./images", help="path to the input images")
    parser.add_argument("--outputs", required=False, default=None, help="path to the output images")
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
        
    # Parse arguments
    args = parse_args()
    # Load config
    load_config(cfg, args.config)
    
    # Create head for post-processing
    head = build_head(cfg['model']['arch']['head'])
    
    # Prepare ONNX engine
    sess = rt.InferenceSession(args.model, providers=rt.get_available_providers())
    input_name = sess.get_inputs()[0].name
    (_, _, height, width) = sess.get_inputs()[0].shape
    assert(height == cfg.data.val.input_size[0] and width == cfg.data.val.input_size[1], 'Size mismatch: {}<>{} or {}<>{}'.format(
        height, cfg.data.val.input_size[0], width, cfg.data.val.input_size[1]
    ))

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
        preds = sess.run(None, {input_name: meta['numpy_img']})
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
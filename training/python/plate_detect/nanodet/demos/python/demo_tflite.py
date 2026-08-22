import argparse, os, cv2, torch, time, tflite_runtime.interpreter as tflite
from demo_utils import get_image_list, load_image
from nanodet.model.head import build_head
from nanodet.util import cfg, load_config

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="model config file path")
    parser.add_argument("--model", required=True, help="model file path")
    parser.add_argument("--inputs", required=True, default="./images", help="path to the input images")
    parser.add_argument("--outputs", required=False, default=None, help="path to the output images")
    parser.add_argument("--delegate", required=False, default=None, help="path to the delegate shared library. E.g. /usr/lib/libvx_delegate.so")
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
    
    # Prepare TFLite engine
    interpreter = tflite.Interpreter(
        args.model,
        experimental_delegates=[tflite.load_delegate(args.delegate)] if args.delegate else None
    )
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    print('input_details:', input_details)
    print('output_details:', output_details)
    
    # Allocate tensors
    interpreter.allocate_tensors()

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
        interpreter.set_tensor(input_details[0]['index'], meta['numpy_img'])
        start_time = time.time()
        interpreter.invoke()
        end_time = time.time()
        preds = interpreter.get_tensor(output_details[0]['index'])
        assert len(preds) == 1, 'We expect a single output'
        # Post-processing (TFLite has no batch size -> expand_dims)
        results = head.post_process(torch.from_numpy(preds[0][None,...]), meta)
        # Visualize and save
        result_img = head.show_result(
            meta["raw_img"][0], results[0], cfg.class_names, score_thres=0.35, show=args.show_result
        )
        print('Inference time: {:2f} ms'.format((end_time - start_time)*1000))
        if args.save_result:
            assert not args.outputs is None, 'Path to --outputs folder must be defined when --save_result is set'
            save_path =  os.path.join(args.outputs, os.path.basename(path))
            cv2.imwrite(save_path, result_img)       
            print('Result saved at {}'.format(save_path))
        if args.show_result:
            ch = cv2.waitKey(0)
            if ch == 27 or ch == ord("q") or ch == ord("Q"):
                break

if __name__ == "__main__":
    main()
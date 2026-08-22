import argparse, os, cv2, torch, time
from hailo_platform import (HEF, VDevice, HailoStreamInterface, InferVStreams, ConfigureParams,
                            InputVStreamParams, OutputVStreamParams, FormatType)
from demo_utils import get_image_list, load_image
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
    
    # Prepare the Hailo inference engine
    hailo_inference = HailoInference(args.model)

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
        print('[{:3d}/{:3d}] Processing {}...'.format(i, len(files), path))
        # Load image without preprocessing or change to NCHW
        # RKNN needs NHWC as input, otherwise get warning suggesting transpose done by the engine
        # Both UINT8 and FLOAT32 are accepted but we noticed that UINT8 is faster. License plate detection (main):
        #   - UINT8: 4 milliseconds
        #   - FLOAT32: 8 milliseconds
        meta = load_image(path, cfg, preprocess=False)       
        # Inference (int8, nhwc)
        start = time.time()
        preds = hailo_inference.run(meta['numpy_img'][0][None,...])
        duration = time.time() - start
        print('Inference time: {} ms'.format(round(duration * 1000)))
        assert len(preds) == 1, 'We expect a single output'
        # Post-processing
        preds_nchw = preds[0].transpose((0, 2, 1)) # hailo produces nhwc output
        results = head.post_process(torch.from_numpy(preds_nchw), meta)
        # Visualize and save
        result_img = head.show_result(
            meta["raw_img"][0], results[0], cfg.class_names, score_thres=0.35, show=args.show_result
        )
        if args.save_result:
            assert not args.outputs is None, 'Path to --outputs folder must be defined when --save_result is set'
            save_path =  os.path.join(args.outputs, os.path.basename(path))
            cv2.imwrite(save_path, result_img)       
            print('Result saved at {}'.format(save_path)) 
        if args.show_result:
            ch = cv2.waitKey(0)
            if ch == 27 or ch == ord("q") or ch == ord("Q"):
                break

# https://raw.githubusercontent.com/serg987/hailo-mini-od-server/d17b8d3838360031f52e7d1d076c92a708c4e786/utils.py
class HailoInference:
    def __init__(self, hef_path, output_type='FLOAT32'):
        """
        Initialize the HailoInference class with the provided HEF model file path.

        Args:
            hef_path (str): Path to the HEF model file.
        """
        self.hef = HEF(hef_path)
        self.target = VDevice()
        self.network_group = self._configure_and_get_network_group()
        self.network_group_params = self.network_group.create_params()
        self.input_vstreams_params, self.output_vstreams_params = self._create_vstream_params(output_type)
        self.input_vstream_info, self.output_vstream_info = self._get_and_print_vstream_info()

    def _configure_and_get_network_group(self):
        """
        Configure the Hailo device and get the network group.

        Returns:
            NetworkGroup: Configured network group.
        """
        configure_params = ConfigureParams.create_from_hef(self.hef, interface=HailoStreamInterface.PCIe)
        network_group = self.target.configure(self.hef, configure_params)[0]
        return network_group

    def _create_vstream_params(self, output_type):
        """
        Create input and output stream parameters.

        Args:
            output_type (str): Format type of the output stream.

        Returns:
            tuple: Input and output stream parameters.
        """
        input_format_type = self.hef.get_input_vstream_infos()[0].format.type
        input_vstreams_params = InputVStreamParams.make_from_network_group(self.network_group, format_type=input_format_type)
        output_vstreams_params = OutputVStreamParams.make_from_network_group(self.network_group, format_type=getattr(FormatType, output_type))
        return input_vstreams_params, output_vstreams_params

    def _get_and_print_vstream_info(self):
        """
        Get and print information about input and output stream layers.

        Returns:
            tuple: List of input stream layer information, List of output stream layer information.
        """
        input_vstream_info = self.hef.get_input_vstream_infos()
        output_vstream_info = self.hef.get_output_vstream_infos()

        for layer_info in input_vstream_info:
            print(f'Input layer: {layer_info.name} {layer_info.shape} {layer_info.format.type}')
        for layer_info in output_vstream_info:
            print(f'Output layer: {layer_info.name} {layer_info.shape} {layer_info.format.type}')

        return input_vstream_info, output_vstream_info

    def get_input_shape(self):
        """
        Get the shape of the model's input layer.

        Returns:
            tuple: Shape of the model's input layer.
        """
        return self.input_vstream_info[0].shape  # Assumes that the model has one input

    def run(self, input_data):
        """
        Run inference on Hailo-8 device.

        Args:
            input_data (np.ndarray, dict, list, tuple): Input data for inference.

        Returns:
            np.ndarray: Inference output.
        """
        input_dict = self._prepare_input_data(input_data)

        with InferVStreams(self.network_group, self.input_vstreams_params, self.output_vstreams_params) as infer_pipeline:
            with self.network_group.activate(self.network_group_params):
                output = infer_pipeline.infer(input_dict)[self.output_vstream_info[0].name]

        return output

    def _prepare_input_data(self, input_data):
        """
        Prepare input data for inference.

        Args:
            input_data (np.ndarray, dict, list, tuple): Input data for inference.

        Returns:
            dict: Prepared input data.
        """
        input_dict = {}
        if isinstance(input_data, dict):
            return input_data
        elif isinstance(input_data, (list, tuple)):
            for layer_info in self.input_vstream_info:
                input_dict[layer_info.name] = input_data
        else:
            if input_data.ndim == 3:
                input_data = np.expand_dims(input_data, axis=0)
            input_dict[self.input_vstream_info[0].name] = input_data

        return input_dict

    def release_device(self):
        """
        Release the Hailo device.
        """
        self.target.release()

if __name__ == "__main__":
    main()
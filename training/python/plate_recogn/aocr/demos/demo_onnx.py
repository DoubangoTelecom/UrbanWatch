import numpy as np
import argparse
import onnxruntime as rt
import cv2
from aocr.config import Config
import os
import json
from rich import print

from demo_utils import get_image_list, load_image, ctc_decode, grid2coords, theta2warp

def demo(cfg, opt):
    
    # Prepare ONNX engine
    model_core = rt.InferenceSession(opt.weights, providers=rt.get_available_providers())
    model_stn = rt.InferenceSession(opt.weights.replace('.', '-stn.'), providers=rt.get_available_providers())
    inp_core = model_core.get_inputs()[0].name
    inp_stn = model_stn.get_inputs()[0].name
    
    # Load demo images
    images_list = get_image_list(opt.images)
    assert len(images_list) > 0, 'Empty image list'

    # alphabet
    alphabet_list = ['$'] + list(cfg.model.alphabet) # $ at 0 is CTC blank
    
    # Ground truth
    with open(os.path.join(opt.images, 'gt.json')) as f:
        ground_truth = json.load(f)

    num_ok = 0
    for image_path in images_list:
        # Load image
        image = load_image(image_path, cfg, preprocess=True)
        
        # STN (TPS or Affine)
        grid_or_theta = model_stn.run(None, {inp_stn: image.reshape((1, 1, cfg.model.imgH, cfg.model.imgW))})
        
        # remap
        if cfg.model.stn_type == 'tps':
            grid_or_theta = grid_or_theta[0].reshape((cfg.model.imgH, cfg.model.imgW, 2)).astype(np.float32)

            dstMap1, dstMap2 = cv2.convertMaps(grid2coords(grid_or_theta, 112, 112), None, cv2.CV_16SC2)
            image_rectified = cv2.remap(image, dstMap1, dstMap2, cv2.INTER_LINEAR, None, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        else:
            assert(cfg.model.stn_type == 'affine')
            M = theta2warp(grid_or_theta[0].reshape((2, 3)), cfg.model.imgW, cfg.model.imgH)
            image_rectified = cv2.warpAffine(image, M, (cfg.model.imgW, cfg.model.imgH))
        
        # visualize
        if opt.visualize:
            cv2.imshow('image', np.concatenate((image, image_rectified), axis=0))
            ch = cv2.waitKey(0)
            if ch == 27 or ch == ord("q") or ch == ord("Q"):
                break
        
        # OCR
        gt = ground_truth[os.path.basename(image_path)]
        preds = model_core.run(None, {inp_core: image_rectified.reshape((1, 1, cfg.model.imgH, cfg.model.imgW))})[0]
        preds_index = np.argmax(preds, axis=-1)
        preds_max_prob = np.take_along_axis(preds, preds_index[...,None], axis=2)
        
        preds_str = ctc_decode(alphabet_list, preds_index.flatten())
        confidence_score = preds_max_prob.flatten().cumprod(axis=0)[-1]
        matched = (gt == preds_str)
        num_ok += 1 if matched else 0
        print('file: {}, pred: {}, score: {}, matched: {}'.format(os.path.basename(image_path), preds_str, confidence_score, matched))
        
    print(':: Accuracy: {:.3f} [{:2}/{:2}]::'.format(num_ok / len(images_list), num_ok, len(images_list)))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True, help="path to config file")
    parser.add_argument('--images', required=True, help='path to the folder containing the images')
    parser.add_argument('--weights', required=True, help="path to models's weigths")
    parser.add_argument('--visualize', action='store_true', help="whether to visualize the rectified image")

    opt = parser.parse_args()
    
    cfg = Config.parse(opt.config)

    demo(cfg, opt)

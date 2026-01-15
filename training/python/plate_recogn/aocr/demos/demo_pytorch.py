import os, time
import argparse

import torch
import torch.backends.cudnn as cudnn
import torch.utils.data

from aocr.config import Config
from aocr.utils import CTCLabelConverter
from aocr.dataset import RawDataset, AlignCollate
from aocr.model import AOCR
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def demo(cfg, opt):
    """ model configuration """
    converter = CTCLabelConverter(cfg.model.alphabet)

    model = AOCR(cfg, training=False).to(device).eval()
    model = torch.nn.DataParallel(model)

    # load model
    print('loading pretrained model from %s' % opt.weights)   
    model.load_state_dict(torch.load(opt.weights, map_location=device), strict=True)

    # prepare data. two demo images from https://github.com/bgshih/crnn#run-demo
    AlignCollate_demo = AlignCollate(imgH=cfg.model.imgH, imgW=cfg.model.imgW, keep_ratio_with_pad=cfg.model.padding)
    demo_data = RawDataset(root=opt.images, opt=cfg)  # use RawDataset
    demo_loader = torch.utils.data.DataLoader(
        demo_data, batch_size=1,
        shuffle=False,
        num_workers=0,
        collate_fn=AlignCollate_demo, pin_memory=True)

    # predict
    model.eval()
    with torch.no_grad():
        for image_tensors, image_path_list in demo_loader:
            batch_size = image_tensors.size(0)
            image = image_tensors.to(device)
            
            # Inference
            preds = model(image)

            # Select max probabilty (greedy decoding) then decode index to character
            preds_size = torch.IntTensor([preds.size(1)] * batch_size)
            preds_max_prob, preds_index = preds.max(-1)
            preds_str = converter.decode(preds_index, preds_size)

            log = open(f'./log_demo_result.txt', 'a')
            dashed_line = '-' * 80
            head = f'{"image_path":25s}\t{"predicted_labels":25s}\tconfidence score'
            
            print(f'{dashed_line}\n{head}\n{dashed_line}')
            log.write(f'{dashed_line}\n{head}\n{dashed_line}\n')

            for img_name, pred, pred_max_prob in zip(image_path_list, preds_str, preds_max_prob):

                # calculate confidence score (= multiply of pred_max_prob)
                confidence_score = pred_max_prob.cumprod(dim=0)[-1]

                print(f'{img_name:25s}\t{pred:25s}\t{confidence_score:0.4f}')
                log.write(f'{img_name:25s}\t{pred:25s}\t{confidence_score:0.4f}\n')

            log.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True, help="path to config file")
    parser.add_argument('--images', required=True, help='path to the folder containing the images')
    parser.add_argument('--weights', required=True, help="path to models's weigths")

    opt = parser.parse_args()
    
    cfg = Config.parse(opt.config)

    cudnn.benchmark = True
    cudnn.deterministic = True

    demo(cfg, opt)

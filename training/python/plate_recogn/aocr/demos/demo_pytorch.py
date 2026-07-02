import os
import argparse

import torch
import torch.backends.cudnn as cudnn
import torch.utils.data

from aocr.config import Config
from aocr.utils import CTCLabelConverter, CELabelConverter
from aocr.dataset import RawDataset, AlignCollate
from aocr.model import AOCR

from rich import print
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def demo(cfg, opt):
    """ model configuration """
    converter = CTCLabelConverter(cfg.model.alphabet) if cfg.train.loss.type == 'ctc' else \
        CELabelConverter(cfg.model.alphabet)

    model = AOCR(cfg, training=False).to(device).eval()

    # load weights
    print('loading pretrained model from %s' % opt.weights)   
    model.load_state_dict(torch.load(opt.weights, map_location=device), strict=True)

    # prepare data. two demo images from https://github.com/bgshih/crnn#run-demo
    AlignCollate_demo = AlignCollate(cfg)
    demo_data = RawDataset(root=opt.images, opt=cfg)  # use RawDataset
    demo_loader = torch.utils.data.DataLoader(
        demo_data, batch_size=1,
        shuffle=False,
        num_workers=0,
        collate_fn=AlignCollate_demo, pin_memory=True)

    # predict
    model.eval()
    num_ok = 0
    num_all = 0
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

            for img_path, pred, pred_max_prob in zip(image_path_list, preds_str, preds_max_prob):

                # calculate confidence score (= multiply of pred_max_prob)
                confidence_score = pred_max_prob.cumprod(dim=0)[-1]
                
                img_name = os.path.basename(img_path)
                gt = img_name.split('.')[0]
                matched = (gt == pred)
                num_ok += 1 if matched else 0
                num_all += 1

                print('{}, pred: {}, score: {:.2}, matched: {}'.format(img_name, pred, confidence_score, matched))

            
    print(':: Accuracy: {:.3f} [{:2}/{:2}]::'.format(num_ok / num_all, num_ok, num_all))

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

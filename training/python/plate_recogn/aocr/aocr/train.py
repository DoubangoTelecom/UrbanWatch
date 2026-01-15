import os
import sys
import time
import random
import copy
import argparse

import torch
import torch.backends.cudnn as cudnn
import torch.nn.init as init
import torch.optim as optim
import torch.utils.data
import numpy as np

from aocr.config import Config
from aocr.utils import CTCLabelConverter, CTCLabelConverterForBaiduWarpctc, AttnLabelConverter, Averager
from aocr.dataset import hierarchical_dataset, AlignCollate, Batch_Balanced_Dataset
from aocr.model import AOCR
from aocr.val import validation
from aocr.focal_loss import FocalLossCTC
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train(cfg):
    # train dataset
    train_dataset = Batch_Balanced_Dataset(cfg)

    # validation dataset
    log = open(f'./saved_models/{cfg.model.name}/log_dataset.txt', 'a')
    AlignCollate_valid = AlignCollate(imgH=cfg.model.imgH, imgW=cfg.model.imgW, keep_ratio_with_pad=cfg.model.padding)
    cfg_val = copy.deepcopy(cfg)
    cfg_val.augment._replace(enabled=False) # no augmentation for validation
    valid_dataset, valid_dataset_log = hierarchical_dataset(root=cfg_val.val.dataset, opt=cfg_val)
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=(cfg_val.train.batch_size*torch.cuda.device_count()),
        shuffle=True,  # 'True' to check training progress with validation function.
        num_workers=int(cfg_val.train.workers*torch.cuda.device_count()),
        collate_fn=AlignCollate_valid, pin_memory=True)
    log.write(valid_dataset_log)
    print('-' * 80)
    log.write('-' * 80 + '\n')
    log.close()
    
    # label converter
    converter = CTCLabelConverter(cfg.model.alphabet)
    
    # Model
    model = AOCR(cfg, training=True).to(device)

    # weight initialization
    for name, param in model.named_parameters():
        if 'localization_fc2' in name or 'loc_fc2' in name or 'fc_loc' in name: # TPS and Affine transformations initialized to identity
            print(f'Skip {name} as it is already initialized')
            continue
        try:
            if 'bias' in name:
                init.constant_(param, 0.0)
            elif 'weight' in name:
                init.kaiming_normal_(param)
        except Exception as e:  # for batchnorm.
            if 'weight' in name:
                param.data.fill_(1)
            continue

    # data parallel for multi-GPU
    model = torch.nn.DataParallel(model)
    model.train()
    model_path = './saved_models/{}/best_norm_ED.pth'.format(cfg.model.name)
    if os.path.exists(model_path):
        print(f'loading pretrained model from {model_path}')
        model.load_state_dict(torch.load(model_path), strict=True)
    else:
        print(f'No weigths at {model_path}')

    """ setup loss """
    criterion = FocalLossCTC(
        **cfg.train.loss.focal_ctc._asdict()
    )
    # loss averager
    loss_avg = Averager()

    # filter that only require gradient decent
    filtered_parameters = []
    params_num = []
    for p in filter(lambda p: p.requires_grad, model.parameters()):
        filtered_parameters.append(p)
        params_num.append(np.prod(p.size()))
    print('Trainable params num : ', sum(params_num))

    # setup optimizer
    if cfg.train.optimizer == 'adam':
        optimizer = optim.Adam(filtered_parameters, lr=cfg.train.optimizer.adam.lr, betas=(cfg.train.optimizer.adam.beta1, 0.999))
    else:
        optimizer = optim.Adadelta(filtered_parameters, lr=cfg.train.optimizer.adadelta.lr, rho=cfg.train.optimizer.adadelta.rho, eps=cfg.train.optimizer.adadelta.eps)
    print("Optimizer:")
    print(optimizer)

    """ final options """
    with open(f'./saved_models/{cfg.model.name}/opt.txt', 'a') as opt_file:
        opt_log = '------------ Options -------------\n'
        args = vars(opt)
        for k, v in args.items():
            opt_log += f'{str(k)}: {str(v)}\n'
        opt_log += '---------------------------------------\n'
        print(opt_log)
        opt_file.write(opt_log)
        
    """ model name """
    print('Model name:', cfg.model.name)
    
    """ start training """
    start_iter = 0
    if model_path != '':
        try:
            start_iter = int(model_path.split('_')[-1].split('.')[0])
            print(f'continue to train, start_iter: {start_iter}')
        except:
            pass

    start_time = time.time()
    best_accuracy = -1
    best_norm_ED = -1
    iteration = start_iter

    while(True):
        
        # train part
        image_tensors, labels = train_dataset.get_batch()
        image = image_tensors
        text, length = converter.encode(labels, batch_max_length=cfg.model.max_len)

        preds = model(image)
        
        preds_size = torch.IntTensor([preds.size(1)] * image.size(0))
        cost = criterion(preds.log_softmax(2).permute(1, 0, 2), text, preds_size, length)

        model.zero_grad()
        cost.backward()
        if cfg.train.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)  # gradient clipping with 5 (Default)
        optimizer.step()

        loss_avg.add(cost)

        # validation part
        if (iteration + 1) % cfg.val.interval == 0 or iteration == 0: # To see training progress, we also conduct validation when 'iteration == 0' 
            elapsed_time = time.time() - start_time
            # for log
            with open(f'./saved_models/{cfg.model.name}/log_train.txt', 'a') as log:
                model.eval()
                with torch.no_grad():
                    valid_loss, current_accuracy, current_norm_ED, preds, confidence_score, labels, infer_time, length_of_data = validation(
                        model, criterion, valid_loader, converter, cfg)
                model.train()

                # training loss and validation loss
                loss_log = f'[{iteration+1}/{cfg.train.num_iter}] Train loss: {loss_avg.val():0.5f}, Valid loss: {valid_loss:0.5f}, Elapsed_time: {elapsed_time:0.5f}'
                loss_avg.reset()

                current_model_log = f'{"Current_accuracy":17s}: {current_accuracy:0.3f}, {"Current_norm_ED":17s}: {current_norm_ED:0.2f}'

                # keep best accuracy model (on valid dataset)
                if current_accuracy > best_accuracy:
                    best_accuracy = current_accuracy
                    torch.save(model.state_dict(), f'./saved_models/{cfg.model.name}/best_accuracy.pth')
                if current_norm_ED > best_norm_ED:
                    best_norm_ED = current_norm_ED
                    torch.save(model.state_dict(), f'./saved_models/{cfg.model.name}/best_norm_ED.pth')
                best_model_log = f'{"Best_accuracy":17s}: {best_accuracy:0.3f}, {"Best_norm_ED":17s}: {best_norm_ED:0.2f}'

                loss_model_log = f'{loss_log}\n{current_model_log}\n{best_model_log}'
                print(loss_model_log)
                log.write(loss_model_log + '\n')

                # show some predicted results
                dashed_line = '-' * 80
                head = f'{"Ground Truth":25s} | {"Prediction":25s} | Confidence Score & T/F'
                predicted_result_log = f'{dashed_line}\n{head}\n{dashed_line}\n'
                for gt, pred, confidence in zip(labels[:5], preds[:5], confidence_score[:5]):
                    predicted_result_log += f'{gt:25s} | {pred:25s} | {confidence:0.4f}\t{str(pred == gt)}\n'
                predicted_result_log += f'{dashed_line}'
                print(predicted_result_log)
                log.write(predicted_result_log + '\n')

        # save model per 1e+5 iter.
        if (iteration + 1) % 1e+5 == 0:
            torch.save(
                model.state_dict(), f'./saved_models/{cfg.model.name}/iter_{iteration+1}.pth')

        if (iteration + 1) == cfg.train.num_iter:
            print('end the training')
            sys.exit()
        iteration += 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config', help="path to config file")
    opt = parser.parse_args()

    # Parse config
    cfg = Config.parse(opt.config)

    os.makedirs(f'./saved_models/{cfg.model.name}', exist_ok=True)

    """ Seed and GPU setting """
    random.seed(cfg.manualSeed)
    np.random.seed(cfg.manualSeed)
    torch.manual_seed(cfg.manualSeed)
    torch.cuda.manual_seed(cfg.manualSeed)

    cudnn.benchmark = True
    cudnn.deterministic = True

    train(cfg)

import torch, torch.optim as optim, os, numpy as np, argparse, copy
from klass.model import KlassModel
from klass.dataset import KlassDataset
from klass.config import Config
from klass.focal_loss import FocalLoss
from klass.averager import Averager

def load_model(model, path, device):
    # Loading existing weights
    if os.path.isfile(path):
        print('Loading existing weigths:', path, ', device:', device)
        model.load_state_dict(torch.load(path, map_location=device),strict=True)
    else:
        print('No weigths found for', path, ', device:', device)
    return model.to(device)

def train(cfg):
    # Check GPU availability
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device_count = max(1, torch.cuda.device_count())
    
    # Create model
    model_folder = 'saved_models'
    os.makedirs(model_folder, exist_ok=True)
    model_path = os.path.join(model_folder, f'{cfg.model.name}.pth')
    model = KlassModel(cfg, training=True)
    model = load_model(model, model_path, device)
    model = torch.nn.DataParallel(model)
        
    # Create datasets
    batch_size = cfg.train.batch_size
    train_dataset = KlassDataset(cfg, 'train')
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=(batch_size*device_count),
        shuffle=True,
        num_workers=int(cfg.train.workers*device_count),
        pin_memory=True)
    cfg_val = copy.deepcopy(cfg)
    cfg_val.augment._replace(enabled=False) # no augmentation for validation
    val_dataset = KlassDataset(cfg_val, 'val')
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=(batch_size*device_count),
        shuffle=True,  # 'True' to check training progress with validation function.
        num_workers=int(cfg_val.train.workers*device_count),
        pin_memory=True)
    
    # Make sure number of classes match
    assert len(train_dataset.class_names) == cfg.model.num_classes, f'Number of classes mismatch: {len(train_dataset.class_names)} <> {cfg.model.num_classes}'
    
    # Loss function and averager
    criterion = FocalLoss(**cfg.train.loss._asdict())
    loss_avg = Averager()
    
    # filter that only require gradient decent
    filtered_parameters = []
    params_num = []
    for p in filter(lambda p: p.requires_grad, model.parameters()):
        filtered_parameters.append(p)
        params_num.append(np.prod(p.size()))
    print('Trainable params num : ', sum(params_num))

    # setup optimizer
    if cfg.train.optimizer.name == 'adam':
        optimizer = optim.Adam(filtered_parameters, lr=cfg.train.optimizer.adam.lr, betas=(cfg.train.optimizer.adam.beta1, 0.999))
    elif cfg.train.optimizer.name == 'sgd':
        optimizer = optim.SGD(filtered_parameters, lr=cfg.train.optimizer.sgd.lr, momentum=cfg.train.optimizer.sgd.momentum, nesterov=cfg.train.optimizer.sgd.nesterov)
    else:
        optimizer = optim.Adadelta(filtered_parameters, lr=cfg.train.optimizer.adadelta.lr, rho=cfg.train.optimizer.adadelta.rho, eps=cfg.train.optimizer.adadelta.eps)
    print("Optimizer:", optimizer)
    
    # LR-Scheduler (https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.StepLR.html#steplr)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=cfg.train.lr_scheduler.step_size, gamma=cfg.train.lr_scheduler.gamma
    )
    
    # Training/Validation loops
    best_epoch = 0
    best_accuracy = 0
    num_epoch_without_accuracy_increase = 0
    correct = 0
    total = 0
    for epoch in range(cfg.train.epochs):
        # Training loop
        model.train(True)
        correct = 0
        total = 0
        loss_avg.reset()
        for batch_idx, (images_, labels_) in enumerate(train_loader):
            images, labels = [x.to(device) for x in [images_, labels_]]
            # inference
            preds = model(images)

            # loss
            cost = criterion(preds.view(-1, preds.shape[-1]), labels.contiguous().view(-1))
            loss_avg.add(cost)

            # backprop
            model.zero_grad()
            cost.backward()
            if cfg.train.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)
            optimizer.step()
            
            # accuracy
            _,pred = torch.max(preds, dim=1)
            correct += torch.sum(pred==labels).item()
            total += labels.size(0)

            # print progress
            print(f'[T#{epoch + 1:5d}/{cfg.train.epochs:5d}/{best_epoch+1:5d}, {batch_idx+1:5d}/{len(train_loader):5d}] loss: {loss_avg.val():.5f}, lr: {lr_scheduler.get_last_lr()[0]:.3e}, acc: {correct / total :.5f}, best_acc: {best_accuracy :.5f}', end='\r')
            
        # End of train loop
        print("", flush=True)
        
        # Validation loop
        with torch.no_grad():
            model.eval()
            correct = 0
            total = 0
            loss_avg.reset()
            for batch_idx, (images_, labels_) in enumerate(val_loader):
                images, labels = [x.to(device) for x in [images_, labels_]]
                preds = model(images)
                cost = criterion(preds.view(-1, preds.shape[-1]), labels.contiguous().view(-1))
                loss_avg.add(cost)
                
                _,pred = torch.max(preds, dim=1)
                correct += torch.sum(pred==labels).item()
                total += labels.size(0)
            
                print(f'[V@{epoch + 1:5d}/{cfg.train.epochs:5d}/{best_epoch+1:5d}, {batch_idx+1:5d}/{len(train_loader):5d}] loss: {loss_avg.val():.5f}, newai:{num_epoch_without_accuracy_increase :2d}, acc: {correct / total :0.5f}, best_acc: {best_accuracy :.5f}', end='\r')
        
            # End of val loop
            print("", flush=True)
        
        # Save best accuracy
        accuracy = correct / total
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            torch.save(model.module.state_dict(), model_path)
            num_epoch_without_accuracy_increase = 0
        else:
            num_epoch_without_accuracy_increase += 1
            if num_epoch_without_accuracy_increase >= cfg.train.lr_scheduler.patience:
                num_epoch_without_accuracy_increase = 0
                lr_scheduler.step()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config', help="path to config file")
    opt = parser.parse_args()

    # Parse config
    cfg = Config.parse(opt.config)
    
    train(cfg)
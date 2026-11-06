import torch, torch.optim as optim, os, argparse
from klass.model import KlassModel
from klass.dataset import KlassDataset
from klass.config import Config
from klass.focal_loss import FocalLoss
from klass.averager import Averager

def load_model(model, path, device):
    # Loading existing weights
    if os.path.isfile(path):
        print('Loading existing weigths:', path, ', device:', device)
        model.load_state_dict(torch.load(path, map_location=device),strict=False)
    else:
        print('No weigths found for', path, ', device:', device)
    return model.to(device)

def train(cfg:Config, reset_best_accuracy:bool = False):
    # Check GPU availability
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device_count = max(1, torch.cuda.device_count())
    
    # Create model
    model_folder = 'saved_models'
    os.makedirs(model_folder, exist_ok=True)
    model_path = os.path.join(model_folder, f'{cfg.model.name}.pth')
    model = KlassModel(cfg, training=True)
    print(f"Backbone: {cfg.model.backbone.type}, Batch size (per device): {cfg.train.batch_size}, Num device: {device_count}, Separator: {cfg.model.feat_separator.enabled}\nOutChannels: {model.backbone.output_channels}, HiddenChannels: {cfg.model.projection.hidden_channels},  NumClasses: {cfg.model.num_classes}")
    model = load_model(model, model_path, device)
    model = torch.nn.DataParallel(model)
        
    # Create datasets
    batch_size = cfg.train.batch_size
    train_dataset = KlassDataset(cfg, 'train')
    train_loader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size=(batch_size*device_count),
        shuffle=True,
        num_workers=int(cfg.train.workers if (cfg.train.workers > 0) else os.cpu_count()),
        pin_memory=True)
    cfg_val = cfg._replace(augment=cfg.augment._replace(enabled=False)) # no augmentation for validation
    val_dataset = KlassDataset(cfg_val, 'val')
    val_loader = torch.utils.data.DataLoader(
        val_dataset, 
        batch_size=(batch_size*device_count),
        shuffle=False,
        num_workers=int(cfg_val.train.workers if (cfg_val.train.workers > 0) else os.cpu_count()),
        pin_memory=True)
    
    # Make sure number of classes match
    assert len(train_dataset.class_names) == cfg.model.num_classes, f'Number of classes mismatch: {len(train_dataset.class_names)} <> {cfg.model.num_classes}'
    
    # Loss function and averager
    criterion = FocalLoss(weight=torch.tensor([cfg.train.loss.alpha,]*cfg.model.num_classes), gamma=cfg.train.loss.gamma)
    loss_avg = Averager()

    # setup optimizer
    optimizer = getattr(optim, cfg.train.optimizer.name)(
        model.parameters(),
        **getattr(cfg.train.optimizer, cfg.train.optimizer.name)._asdict()
    )
    print("Optimizer:", optimizer)
    
    # LR-Scheduler
    lr_scheduler = getattr(torch.optim.lr_scheduler, cfg.train.lr_scheduler.name)(
        optimizer,
        **getattr(cfg.train.lr_scheduler, cfg.train.lr_scheduler.name)._asdict()
    )
    print("Scheduler:", lr_scheduler)

    ## Validation function ##
    def validation_loop(epoch:int = 0, best_accuracy_val:float = 0):
        with torch.no_grad():
            model.eval()
            correct = 0
            total = 0
            loss_avg.reset()
            for batch_idx, (images_, labels_) in enumerate(val_loader):
                images, labels = [x.to(device) for x in [images_, labels_]]
                preds, feat_logits = model(images, labels)
                cost = torch.tensor(0.).to(preds.device)
                for p in [preds, feat_logits]:
                    if not p is None:
                        cost += criterion(p.view(-1, p.shape[-1]), labels.contiguous().view(-1))
                loss_avg.add(cost)
                
                _,pred = torch.max(preds, dim=1)
                correct += torch.sum(pred==labels).item()
                total += labels.size(0)

                accuracy = correct / total
                print(f'[{cfg.model.name}] [V@{epoch + 1:5d}/{cfg.train.epochs:5d}, {batch_idx+1:5d}/{len(val_loader):5d}] loss: {loss_avg.val():.5f}, newai:{num_epoch_without_accuracy_increase :2d}/{cfg.train.lr_scheduler.patience :2d}, acc: {correct / total :0.5f}, best_acc_val: {max(accuracy, best_accuracy_val) :.5f}', end='\r')

        # End of train loop
        print("", flush=True)

        return accuracy

    ## Training function ##
    def training_loop(epoch:int = 0, best_accuracy_val:float = 0):
        model.train(True)
        correct = 0
        total = 0
        loss_avg.reset()
        for batch_idx, (images_, labels_) in enumerate(train_loader):
            images, labels = [x.to(device) for x in [images_, labels_]]

            # inference
            preds, feat_logits = model(images, labels)

            # loss (+L2-regul)
            cost = torch.tensor(0.).to(preds.device)
            for p in [preds, feat_logits]:
                if not p is None:
                    cost += criterion(p.view(-1, p.shape[-1]), labels.contiguous().view(-1))
            if cfg.train.l2_lambda > 0:
                l2_reg = torch.tensor(0.).to(cost.device)
                for param in model.parameters():
                    l2_reg += torch.norm(param)
                cost += cfg.train.l2_lambda * l2_reg
            loss_avg.add(cost)

            # backprop
            cost.backward()
            if cfg.train.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)
            optimizer.step()
            optimizer.zero_grad()
            
            # accuracy
            _,pred = torch.max(preds, dim=1)
            correct += torch.sum(pred==labels).item()
            total += labels.size(0)

            # print progress
            accuracy = correct / total
            print(f'[{cfg.model.name}] [T#{epoch + 1:5d}/{cfg.train.epochs:5d}, {batch_idx+1:5d}/{len(train_loader):5d}] loss: {loss_avg.val():.5f}, lr: {lr_scheduler.get_last_lr()[0]:.3e}, newai:{num_epoch_without_accuracy_increase :2d}/{cfg.train.lr_scheduler.patience :2d}, acc: {accuracy :.5f}, best_acc_val: {best_accuracy_val :.5f}', end='\r')
            
        # End of train loop
        print("", flush=True)

        return accuracy
    
    # Training/Validation loops
    num_epoch_without_accuracy_increase = 0
    best_accuracy_val = validation_loop() if (not reset_best_accuracy and os.path.isfile(model_path)) else 0
    best_accuracy_train = 0
    for epoch in range(cfg.train.epochs):
        accuracy_train = training_loop(epoch, best_accuracy_val)
        accuracy_val = validation_loop(epoch, best_accuracy_val)
        
        # Save best accuracy
        if accuracy_val > best_accuracy_val or accuracy_train > best_accuracy_train:
            best_accuracy_train = max(accuracy_train, best_accuracy_train)
            if accuracy_val > best_accuracy_val:
                torch.save(model.module.state_dict(), model_path)
                best_accuracy_val = accuracy_val
            num_epoch_without_accuracy_increase = 0
        else:
            num_epoch_without_accuracy_increase += 1
            if num_epoch_without_accuracy_increase >= cfg.train.lr_scheduler.patience:
                num_epoch_without_accuracy_increase = 0
                lr_scheduler.step()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config', help="path to config file")
    parser.add_argument("--reset", action=argparse.BooleanOptionalAction, required=False, default=False, help="Whether to reset the best accuracy")
    opt = parser.parse_args()

    # Parse config
    cfg = Config.parse(opt.config)
    
    train(cfg, opt.reset)
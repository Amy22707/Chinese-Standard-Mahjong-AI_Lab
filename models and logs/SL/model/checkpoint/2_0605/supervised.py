from dataset import MahjongGBDataset
from torch.utils.data import DataLoader
from model import CNNModel
import torch.nn.functional as F
import torch
import argparse
import os


def get_device(preferred):
    if preferred == 'auto':
        if hasattr(torch, 'npu') and torch.npu.is_available():
            return torch.device('npu')
        if torch.cuda.is_available():
            return torch.device('cuda')
        return torch.device('cpu')
    return torch.device(preferred)


def move_batch(batch, device):
    obs, mask, act = batch
    # non_blocking is only safe on CUDA; disable on NPU to avoid task-scheduler errors
    non_blocking = device.type == 'cuda'
    return {
        'observation': obs.to(device, non_blocking = non_blocking),
        'action_mask': mask.to(device, non_blocking = non_blocking)
    }, act.long().to(device, non_blocking = non_blocking)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type = int, default = 32)
    parser.add_argument('--batch-size', type = int, default = 1024)
    parser.add_argument('--lr', type = float, default = 1e-3)
    parser.add_argument('--min-lr', type = float, default = 1e-5)
    parser.add_argument('--weight-decay', type = float, default = 1e-4)
    parser.add_argument('--label-smoothing', type = float, default = 0.05)
    parser.add_argument('--split-ratio', type = float, default = 0.9)
    parser.add_argument('--num-workers', type = int, default = 0)
    parser.add_argument('--device', default = 'auto')
    parser.add_argument('--logdir', default = 'model')
    parser.add_argument('--resume', type = str, default = None,
                        help = 'path to an epoch checkpoint (.pkl) to resume training from')
    return parser.parse_args()
 
if __name__ == '__main__':
    args = parse_args()
    checkpoint_dir = os.path.join(args.logdir, 'checkpoint')
    os.makedirs(checkpoint_dir, exist_ok = True)
    device = get_device(args.device)
    
    # Load dataset
    trainDataset = MahjongGBDataset(0, args.split_ratio, True)
    validateDataset = MahjongGBDataset(args.split_ratio, 1, False)
    # pin_memory is only beneficial on CUDA; disable on NPU
    pin_memory = device.type == 'cuda'
    loader = DataLoader(dataset = trainDataset, batch_size = args.batch_size, shuffle = True,
                        num_workers = args.num_workers, pin_memory = pin_memory)
    vloader = DataLoader(dataset = validateDataset, batch_size = args.batch_size, shuffle = False,
                         num_workers = args.num_workers, pin_memory = pin_memory)
    
    # Load model
    model = CNNModel().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr = args.lr, weight_decay = args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = args.epochs, eta_min = args.min_lr)
    best_acc = 0
    start_epoch = 0

    # Resume from checkpoint if requested
    if args.resume:
        ckpt = torch.load(args.resume, map_location = 'cpu')
        if isinstance(ckpt, dict) and 'model' in ckpt:
            # Full training checkpoint
            model.load_state_dict(ckpt['model'])
            optimizer.load_state_dict(ckpt['optimizer'])
            scheduler.load_state_dict(ckpt['scheduler'])
            start_epoch = ckpt['epoch'] + 1
            best_acc = ckpt.get('best_acc', 0)
            print('Resumed from epoch %d, best_acc=%.4f' % (ckpt['epoch'], best_acc))
        else:
            # Legacy: plain state dict (e.g. best.pkl)
            model.load_state_dict(ckpt)
            print('Loaded model weights from %s (no optimizer state)' % args.resume)
        model.to(device)
    
    # Train and validate
    for e in range(start_epoch, args.epochs):
        print('Epoch', e)
        model.train(True)
        for i, d in enumerate(loader):
            input_dict, target = move_batch(d, device)
            logits = model(input_dict)
            loss = F.cross_entropy(logits, target, label_smoothing = args.label_smoothing)
            if i % 128 == 0:
                print('Iteration %d/%d'%(i, len(trainDataset) // args.batch_size + 1), 'policy_loss', loss.item())
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()
        scheduler.step()
        print('Run validation:')
        correct = 0
        total_loss = 0
        model.train(False)
        for i, d in enumerate(vloader):
            input_dict, target = move_batch(d, device)
            with torch.no_grad():
                logits = model(input_dict)
                pred = logits.argmax(dim = 1)
                total_loss += F.cross_entropy(logits, target, reduction = 'sum').item()
                correct += torch.eq(pred, target).sum().item()
        acc = correct / len(validateDataset)
        val_loss = total_loss / len(validateDataset)
        cpu_state_dict = {k: v.cpu() for k, v in model.state_dict().items()}
        # Save full checkpoint for resumption
        torch.save({
            'epoch'     : e,
            'model'     : cpu_state_dict,
            'optimizer' : optimizer.state_dict(),
            'scheduler' : scheduler.state_dict(),
            'best_acc'  : best_acc,
        }, os.path.join(checkpoint_dir, '%d.pkl' % e))
        if acc > best_acc:
            best_acc = acc
            # best.pkl is a plain state dict so __main__.py can load it directly
            torch.save(cpu_state_dict, os.path.join(checkpoint_dir, 'best.pkl'))
        print('Epoch', e + 1, 'Validate loss:', val_loss, 'Validate acc:', acc, 'Best acc:', best_acc)

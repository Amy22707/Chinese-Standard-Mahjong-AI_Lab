from dataset import MahjongGBDataset
from torch.utils.data import DataLoader, WeightedRandomSampler
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
    obs, mask, act, wt, aux = batch
    # non_blocking is only safe on CUDA; disable on NPU to avoid task-scheduler errors
    non_blocking = device.type == 'cuda'
    input_dict = {
        'observation': obs.to(device, non_blocking = non_blocking),
        'action_mask': mask.to(device, non_blocking = non_blocking)
    }
    aux = {k: v.float().to(device, non_blocking = non_blocking) for k, v in aux.items()}
    return input_dict, act.long().to(device, non_blocking = non_blocking), wt.float().to(device, non_blocking = non_blocking), aux


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type = int, default = 32)
    parser.add_argument('--batch-size', type = int, default = 1024)
    parser.add_argument('--lr', type = float, default = 1e-3)
    parser.add_argument('--min-lr', type = float, default = 1e-5)
    parser.add_argument('--weight-decay', type = float, default = 1e-4)
    parser.add_argument('--label-smoothing', type = float, default = 0.0)
    parser.add_argument('--type-loss-weight', type = float, default = 0.2)
    parser.add_argument('--win-loss-weight', type = float, default = 0.05)
    parser.add_argument('--fan-loss-weight', type = float, default = 0.10)
    parser.add_argument('--shanten-loss-weight', type = float, default = 0.05)
    parser.add_argument('--discard-rank-loss-weight', type = float, default = 0.10)
    parser.add_argument('--weighted-sampler', action = 'store_true',
                        help = 'oversample high-weight/high-score samples instead of only weighting the loss')
    parser.add_argument('--split-ratio', type = float, default = 0.9)
    parser.add_argument('--num-workers', type = int, default = 0)
    parser.add_argument('--cache-size', type = int, default = 2048,
                        help = 'number of match .npz files cached per DataLoader worker')
    parser.add_argument('--prefetch-factor', type = int, default = 2,
                        help = 'DataLoader prefetch factor when num-workers > 0')
    parser.add_argument('--no-persistent-workers', action = 'store_true',
                        help = 'disable persistent DataLoader workers')
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
    trainDataset = MahjongGBDataset(0, args.split_ratio, True, cache_size = args.cache_size)
    validateDataset = MahjongGBDataset(args.split_ratio, 1, False, cache_size = args.cache_size)
    # pin_memory is only beneficial on CUDA; disable on NPU
    pin_memory = device.type == 'cuda'
    sampler = None
    shuffle = True
    if args.weighted_sampler:
        sampler = WeightedRandomSampler(
            weights = torch.from_numpy(trainDataset.sample_weights()).double(),
            num_samples = len(trainDataset),
            replacement = True
        )
        shuffle = False
    loader_kwargs = {
        'num_workers': args.num_workers,
        'pin_memory': pin_memory,
    }
    if args.num_workers > 0:
        loader_kwargs['prefetch_factor'] = args.prefetch_factor
        loader_kwargs['persistent_workers'] = not args.no_persistent_workers
    loader = DataLoader(dataset = trainDataset, batch_size = args.batch_size, shuffle = shuffle,
                        sampler = sampler, **loader_kwargs)
    vloader = DataLoader(dataset = validateDataset, batch_size = args.batch_size, shuffle = False,
                         **loader_kwargs)
    
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
            print('Resumed from epoch %d, best_acc=%.4f' % (ckpt['epoch'], best_acc), flush = True)
        else:
            # Legacy: plain state dict (e.g. best.pkl)
            model.load_state_dict(ckpt)
            print('Loaded model weights from %s (no optimizer state)' % args.resume, flush = True)
        model.to(device)
    
    # Train and validate
    for e in range(start_epoch, args.epochs):
        print('Epoch', e, flush = True)
        model.train(True)
        for i, d in enumerate(loader):
            input_dict, target, wt, aux_target = move_batch(d, device)
            logits, type_logits, aux_pred = model(input_dict, return_type_logits = True, return_aux = True)
            # Weighted policy loss: higher-fan games contribute more
            per_sample_loss = F.cross_entropy(logits, target,
                                              label_smoothing = args.label_smoothing,
                                              reduction = 'none')
            wt_norm = wt / wt.mean().clamp(min = 1e-6)  # normalize so mean weight ~1
            policy_loss = (per_sample_loss * wt_norm).mean()
            type_target = model.action_type_targets(target)
            type_loss = F.cross_entropy(type_logits, type_target)
            win_loss = F.binary_cross_entropy_with_logits(aux_pred['win_logit'], aux_target['win'])
            fan_loss = F.smooth_l1_loss(aux_pred['fan'].sigmoid(), aux_target['fan'])
            shanten_loss = F.smooth_l1_loss(aux_pred['shanten'].sigmoid(), aux_target['shanten'])
            discard_mask = aux_target['discard_rank'] >= 0
            if discard_mask.any():
                discard_rank_loss = F.mse_loss(
                    aux_pred['discard_rank'].sigmoid()[discard_mask],
                    aux_target['discard_rank'][discard_mask]
                )
            else:
                discard_rank_loss = logits.new_tensor(0.0)
            loss = (policy_loss
                    + args.type_loss_weight * type_loss
                    + args.win_loss_weight * win_loss
                    + args.fan_loss_weight * fan_loss
                    + args.shanten_loss_weight * shanten_loss
                    + args.discard_rank_loss_weight * discard_rank_loss)
            if i % 128 == 0:
                print('Iteration %d/%d'%(i, len(trainDataset) // args.batch_size + 1),
                      'policy_loss', policy_loss.item(), 'type_loss', type_loss.item(),
                      'win_loss', win_loss.item(), 'fan_loss', fan_loss.item(),
                      'shanten_loss', shanten_loss.item(), 'discard_rank_loss', discard_rank_loss.item(),
                      flush = True)
            optimizer.zero_grad(set_to_none = True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()
        scheduler.step()
        print('Run validation:', flush = True)
        correct = 0
        total_loss = 0
        model.train(False)
        for i, d in enumerate(vloader):
            input_dict, target, _, _ = move_batch(d, device)
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
        print('Epoch', e + 1, 'Validate loss:', val_loss, 'Validate acc:', acc, 'Best acc:', best_acc, flush = True)

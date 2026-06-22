from multiprocessing import Process, Event
import time
import numpy as np
import torch
from torch.nn import functional as F
import signal
import os

from replay_buffer import ReplayBuffer
from model_pool import ModelPoolServer
from model import CNNModel

def _npu_available():
    try:
        import torch_npu  # noqa: F401
    except ImportError:
        return False
    return hasattr(torch, 'npu') and hasattr(torch.npu, 'is_available') and torch.npu.is_available()

def _try_make_writer(log_dir):
    '''Return a SummaryWriter or None if TensorBoard is unavailable.'''
    try:
        from torch.utils.tensorboard import SummaryWriter
        os.makedirs(log_dir, exist_ok=True)
        return SummaryWriter(log_dir=log_dir)
    except Exception:
        return None

class Learner(Process):
    
    def __init__(self, config, replay_buffer):
        super(Learner, self).__init__()
        self.replay_buffer = replay_buffer
        self.config = config
        self.stop_event = Event()

    def stop(self):
        self.stop_event.set()
    
    def run(self):
        # keep signal handling in main process only
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        signal.signal(signal.SIGTERM, lambda signum, frame: self.stop_event.set())
        # create model pool
        model_pool = ModelPoolServer(self.config['model_pool_size'], self.config['model_pool_name'])
        try:
            # ── Device selection ─────────────────────────────────────────────
            requested_device = str(self.config.get('device', 'cpu')).strip().lower()
            if requested_device == 'auto':
                if _npu_available():
                    requested_device = 'npu'
                elif torch.cuda.is_available():
                    requested_device = 'cuda'
                else:
                    requested_device = 'cpu'
            if requested_device.startswith('npu') and not _npu_available():
                print('Warning: requested NPU device but NPU is unavailable, using CPU in learner.')
                requested_device = 'cpu'
            if requested_device.startswith('cuda') and not torch.cuda.is_available():
                print('Warning: requested CUDA device but CUDA is unavailable, using CPU in learner.')
                requested_device = 'cpu'
            device = torch.device(requested_device)

            # ── Build model ──────────────────────────────────────────────────
            model = CNNModel()
            teacher_model = None

            # Warm-start from SL checkpoint if provided
            sl_ckpt = self.config.get('sl_checkpoint', '')
            if sl_ckpt and os.path.isfile(sl_ckpt):
                model.load_sl_checkpoint(sl_ckpt, device='cpu')
                if self.config.get('kl_coeff_start', 0.0) > 0:
                    teacher_model = CNNModel()
                    teacher_model.load_sl_checkpoint(sl_ckpt, device='cpu')
                    teacher_model = teacher_model.to(device)
                    teacher_model.eval()
                    for p in teacher_model.parameters():
                        p.requires_grad_(False)
                    print('[Learner] Enabled SL KL regularisation with teacher %s.' % sl_ckpt)
            elif self.config.get('kl_coeff_start', 0.0) > 0:
                print('[Learner] Warning: KL coefficient is enabled but sl_checkpoint is missing; KL regularisation disabled.')

            # Resume from RL checkpoint if provided
            resume_ckpt = self.config.get('resume_checkpoint', '')
            start_iter = 0
            if resume_ckpt and os.path.isfile(resume_ckpt):
                ckpt = torch.load(resume_ckpt, map_location='cpu')
                model.load_state_dict(ckpt['model'])
                start_iter = ckpt.get('iteration', 0)
                print('[Learner] Resumed from %s at iteration %d' % (resume_ckpt, start_iter))

            # Push initial weights to model pool (CPU tensors required)
            model_pool.push(model.state_dict())
            model = model.to(device)

            # ── Optimizer and LR scheduler ───────────────────────────────────
            optimizer = torch.optim.Adam(model.parameters(), lr=self.config['lr'])
            total_updates = self.config.get('total_updates', 100000)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=total_updates, eta_min=self.config['lr'] * 0.01
            )
            if resume_ckpt and os.path.isfile(resume_ckpt):
                # ckpt already loaded above �?reuse without re-reading file
                if 'optimizer' in ckpt:
                    optimizer.load_state_dict(ckpt['optimizer'])
                if 'scheduler' in ckpt:
                    scheduler.load_state_dict(ckpt['scheduler'])

            # ── TensorBoard writer (optional) ────────────────────────────────
            tb_dir = self.config.get('tensorboard_dir', '')
            writer = _try_make_writer(tb_dir) if tb_dir else None

            # ── Wait for initial samples ─────────────────────────────────────
            while self.replay_buffer.size() < self.config['min_sample'] and not self.stop_event.is_set():
                time.sleep(0.1)

            cur_time = time.time()
            iterations = start_iter
            max_grad_norm = self.config.get('max_grad_norm', 0.5)

            while not self.stop_event.is_set():
                # ── Sample a batch ───────────────────────────────────────────
                batch = self.replay_buffer.sample(self.config['batch_size'])
                obs    = torch.tensor(batch['state']['observation']).to(device)
                mask   = torch.tensor(batch['state']['action_mask']).to(device)
                states = {'observation': obs, 'action_mask': mask}
                actions   = torch.tensor(batch['action']).unsqueeze(-1).to(device)
                advs      = torch.tensor(batch['adv']).to(device)
                targets   = torch.tensor(batch['target']).to(device)

                # old_values: critic estimates at collection time, used for value clipping
                old_values    = torch.tensor(batch['value']).to(device)
                # old_log_probs: policy log-probs at collection time, used for PPO ratio
                old_log_probs = torch.tensor(batch['log_prob']).unsqueeze(-1).to(device)

                # Normalise advantages for training stability
                advs = (advs - advs.mean()) / (advs.std() + 1e-8)

                print('Iteration %d, replay buffer in %d out %d' % (
                    iterations,
                    self.replay_buffer.stats['sample_in'],
                    self.replay_buffer.stats['sample_out']
                ))

                # ── PPO update epochs ────────────────────────────────────────
                total_policy_loss = 0.0
                total_value_loss  = 0.0
                total_entropy     = 0.0
                total_kl_loss     = 0.0
                clip_range = self.config['clip']
                kl_coeff = self._current_kl_coeff(iterations)
                target_kl = self.config.get('target_kl', 0.02)  # early-stop threshold
                model.train(True)  # BatchNorm training mode
                epochs_done = 0
                for epoch in range(self.config['epochs']):
                    logits, values_pred = model(states)
                    action_dist = torch.distributions.Categorical(logits=logits)

                    # Clipped surrogate objective
                    log_probs = action_dist.log_prob(actions.squeeze(-1)).unsqueeze(-1)
                    ratio     = torch.exp(log_probs - old_log_probs)
                    surr1     = ratio * advs.unsqueeze(-1)
                    surr2     = torch.clamp(ratio, 1 - clip_range, 1 + clip_range) * advs.unsqueeze(-1)
                    policy_loss = -torch.mean(torch.min(surr1, surr2))

                    # Value loss.  An optional value_clip_range (much larger than policy clip_range)
                    # prevents sudden large value jumps without over-constraining learning.
                    # When value_clip_range <= 0 the clipping is disabled (plain MSE, recommended).
                    values_pred = values_pred.squeeze(-1)
                    value_clip_range = float(self.config.get('value_clip_range', 0.0))
                    if value_clip_range > 0:
                        values_clipped = old_values + torch.clamp(
                            values_pred - old_values, -value_clip_range, value_clip_range
                        )
                        value_loss = torch.mean(torch.max(
                            F.mse_loss(values_pred,    targets, reduction='none'),
                            F.mse_loss(values_clipped, targets, reduction='none'),
                        ))
                    else:
                        value_loss = F.mse_loss(values_pred, targets)

                    # Entropy bonus to encourage exploration
                    entropy_loss = -torch.mean(action_dist.entropy())

                    # KL constraint to keep RL policy close to the frozen SL teacher early on.
                    if teacher_model is not None and kl_coeff > 0:
                        with torch.no_grad():
                            teacher_logits, _ = teacher_model(states)
                            teacher_probs = F.softmax(teacher_logits, dim=-1)
                        kl_loss = F.kl_div(
                            F.log_softmax(logits, dim=-1),
                            teacher_probs,
                            reduction='batchmean'
                        )
                    else:
                        kl_loss = logits.new_tensor(0.0)

                    loss = (policy_loss
                            + self.config['value_coeff']   * value_loss
                            + self.config['entropy_coeff'] * entropy_loss
                            + kl_coeff * kl_loss)

                    optimizer.zero_grad()
                    loss.backward()
                    # Gradient clipping to prevent exploding gradients
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                    optimizer.step()

                    total_policy_loss += policy_loss.item()
                    total_value_loss  += value_loss.item()
                    total_entropy     += -entropy_loss.item()
                    total_kl_loss     += kl_loss.item()
                    epochs_done += 1

                    # PPO early stopping: if policy has drifted too far from old policy,
                    # further epochs are unreliable (importance ratios are stale).
                    with torch.no_grad():
                        approx_kl = 0.5 * ((log_probs.detach() - old_log_probs) ** 2).mean().item()
                    if approx_kl > 1.5 * target_kl:
                        break

                scheduler.step()

                # ── Push updated model to pool (CPU tensors) ─────────────────
                model_pool.push({k: v.detach().cpu() for k, v in model.state_dict().items()})

                # ── TensorBoard logging ──────────────────────────────────────
                if writer is not None:
                    n = epochs_done
                    writer.add_scalar('loss/policy',  total_policy_loss / n, iterations)
                    writer.add_scalar('loss/value',   total_value_loss  / n, iterations)
                    writer.add_scalar('loss/entropy', total_entropy     / n, iterations)
                    writer.add_scalar('loss/kl_sl',   total_kl_loss     / n, iterations)
                    writer.add_scalar('train/kl_coeff', kl_coeff, iterations)
                    writer.add_scalar('train/epochs_done', epochs_done, iterations)
                    writer.add_scalar('train/lr',     scheduler.get_last_lr()[0], iterations)
                    writer.add_scalar('buffer/in',    self.replay_buffer.stats['sample_in'],  iterations)
                    writer.add_scalar('buffer/out',   self.replay_buffer.stats['sample_out'], iterations)

                # ── Periodic checkpoint save ─────────────────────────────────
                t = time.time()
                if t - cur_time > self.config['ckpt_save_interval']:
                    ckpt_path = os.path.join(self.config['ckpt_save_path'], 'model_%d.pt' % iterations)
                    torch.save({
                        'model'     : {k: v.detach().cpu() for k, v in model.state_dict().items()},
                        'optimizer' : optimizer.state_dict(),
                        'scheduler' : scheduler.state_dict(),
                        'iteration' : iterations,
                    }, ckpt_path)
                    print('[Learner] Saved checkpoint to %s' % ckpt_path)
                    cur_time = t

                iterations += 1
        finally:
            if writer is not None:
                writer.close()
            model_pool.close()

    def _current_kl_coeff(self, iteration):
        '''Linearly anneal SL KL regularisation from start to end coefficient.'''
        start = float(self.config.get('kl_coeff_start', 0.0))
        end = float(self.config.get('kl_coeff_end', start))
        steps = max(1, int(self.config.get('kl_anneal_updates', 1)))
        ratio = min(1.0, max(0.0, iteration / steps))
        return start + (end - start) * ratio

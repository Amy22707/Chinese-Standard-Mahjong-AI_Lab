import argparse
from replay_buffer import ReplayBuffer
from actor import Actor
from learner import Learner
import signal
import torch
import os
import time

def _npu_available():
    try:
        import torch_npu  # noqa: F401
    except ImportError:
        return False
    return hasattr(torch, 'npu') and hasattr(torch.npu, 'is_available') and torch.npu.is_available()

def _parse_args():
    parser = argparse.ArgumentParser(description='Chinese Standard Mahjong RL Training')
    parser.add_argument('--device', type=str, default='auto',
                        help='Training device: auto | cpu | cuda | npu (default: auto)')
    parser.add_argument('--num_actors', type=int, default=24,
                        help='Number of parallel actor processes (default: 24)')
    parser.add_argument('--episodes_per_actor', type=int, default=1000,
                        help='Episodes each actor collects before stopping (default: 1000)')
    parser.add_argument('--total_updates', type=int, default=100000,
                        help='Total learner update steps for LR scheduler (default: 100000)')
    parser.add_argument('--sl_checkpoint', type=str, default='',
                        help='Path to SL .pt checkpoint for warm-starting the policy (default: none)')
    parser.add_argument('--resume', type=str, default='',
                        dest='resume_checkpoint',
                        help='Path to RL checkpoint to resume training from (default: none)')
    parser.add_argument('--checkpoint_dir', type=str, default='',
                        help='Directory to save RL checkpoints (default: <script_dir>/checkpoint/)')
    parser.add_argument('--tensorboard_dir', type=str, default='',
                        help='Directory for TensorBoard logs (default: none, logging disabled)')
    parser.add_argument('--lr', type=float, default=3e-5, help='Learning rate (default: 3e-5, recommended for SL warm-start)')
    parser.add_argument('--gamma', type=float, default=0.98, help='Discount factor γ (default: 0.98)')
    parser.add_argument('--lam', type=float, default=0.95, help='GAE λ parameter (default: 0.95)')
    parser.add_argument('--clip', type=float, default=0.2, help='PPO clip range ε (default: 0.2)')
    parser.add_argument('--value_clip_range', type=float, default=0.0,
                        help='Value loss clip range; 0=plain MSE (recommended). If set, use a '
                             'value >= max reward (e.g. 50.0), NOT the PPO policy clip (default: 0.0)')
    parser.add_argument('--epochs', type=int, default=5, help='PPO update epochs per batch (default: 5)')
    parser.add_argument('--batch_size', type=int, default=256, help='Minibatch size (default: 256)')
    parser.add_argument('--kl_coeff_start', type=float, default=0.5,
                        help='Initial SL teacher KL coefficient (default: 0.5)')
    parser.add_argument('--kl_coeff_end', type=float, default=0.01,
                        help='Final SL teacher KL coefficient after annealing (default: 0.01)')
    parser.add_argument('--kl_anneal_updates', type=int, default=100000,
                        help='Learner updates over which KL coefficient is linearly annealed (default: 100000)')
    parser.add_argument('--max_grad_norm', type=float, default=0.5,
                        help='Max gradient norm for clipping (default: 0.5)')
    parser.add_argument('--reward_scale', type=float, default=10.0,
                        help='Divide all env rewards by this factor (default: 10.0, normalises win rewards to ~3-30)')
    parser.add_argument('--reward_tenpai', type=float, default=2.0,
                        help='Tenpai bonus at Huang (draw game) (default: 2.0)')
    parser.add_argument('--reward_notenpai', type=float, default=-2.0,
                        help='Penalty for not being in tenpai at Huang (default: -2.0)')
    return parser.parse_args()


if __name__ == '__main__':
    args = _parse_args()
    base_dir = os.path.dirname(os.path.abspath(__file__))

    ckpt_dir = args.checkpoint_dir or os.path.join(base_dir, 'checkpoint')
    os.makedirs(ckpt_dir, exist_ok=True)

    # ── Device resolution ────────────────────────────────────────────────────
    device = args.device
    if device == 'auto':
        if _npu_available():
            device = 'npu'
        elif torch.cuda.is_available():
            device = 'cuda'
        else:
            device = 'cpu'
    elif device.startswith('npu') and not _npu_available():
        print('Warning: NPU is not available, fallback to CPU.')
        device = 'cpu'
    elif device.startswith('cuda') and not torch.cuda.is_available():
        print('Warning: CUDA is not available in current PyTorch build, fallback to CPU.')
        device = 'cpu'

    # ── Environment config (passed to actors) ────────────────────────────────
    from feature import FeatureAgent
    env_config = {
        'agent_clz'      : FeatureAgent,
        'reward_scale'   : args.reward_scale,
        'reward_tenpai'  : args.reward_tenpai,
        'reward_notenpai': args.reward_notenpai,
    }

    # ── Main training config ─────────────────────────────────────────────────
    config = {
        # Infrastructure
        'replay_buffer_size'   : 50000,
        'replay_buffer_episode': 400,
        'model_pool_size'      : 20,
        'model_pool_name'      : 'model-pool',
        # Actor
        'num_actors'           : args.num_actors,
        'episodes_per_actor'   : args.episodes_per_actor,
        'env_config'           : env_config,
        # PPO / GAE
        'gamma'                : args.gamma,
        'lambda'               : args.lam,
        'clip'                 : args.clip,
        'value_clip_range'     : args.value_clip_range,
        'epochs'               : args.epochs,
        'batch_size'           : args.batch_size,
        'min_sample'           : 200,
        'value_coeff'          : 0.5,
        'entropy_coeff'        : 0.01,
        'kl_coeff_start'       : args.kl_coeff_start,
        'kl_coeff_end'         : args.kl_coeff_end,
        'kl_anneal_updates'    : args.kl_anneal_updates,
        'max_grad_norm'        : args.max_grad_norm,
        # Optimiser
        'lr'                   : args.lr,
        'total_updates'        : args.total_updates,
        # Device & checkpoints
        'device'               : device,
        'ckpt_save_interval'   : 300,   # seconds between checkpoint saves
        'ckpt_save_path'       : ckpt_dir + os.sep,
        # Optional warm-start and resume
        'sl_checkpoint'        : args.sl_checkpoint,
        'resume_checkpoint'    : args.resume_checkpoint,
        # Logging
        'tensorboard_dir'      : args.tensorboard_dir,
    }

    replay_buffer = ReplayBuffer(config['replay_buffer_size'], config['replay_buffer_episode'])

    actors = []
    for i in range(config['num_actors']):
        cfg = dict(config)
        cfg['name'] = 'Actor-%d' % i
        actor = Actor(cfg, replay_buffer)
        actors.append(actor)
    learner = Learner(config, replay_buffer)

    stop_requested = {'value': False}

    def _request_stop(signum, frame):
        if not stop_requested['value']:
            print('Received signal %d, shutting down...' % signum)
        stop_requested['value'] = True

    signal.signal(signal.SIGINT, _request_stop)
    signal.signal(signal.SIGTERM, _request_stop)

    def _wait_processes(processes, timeout):
        alive = [p for p in processes if p.is_alive()]
        deadline = time.time() + timeout
        while alive and time.time() < deadline:
            for p in alive[:]:
                p.join(timeout=0.1)
                if not p.is_alive():
                    alive.remove(p)
        return alive

    started_actors = []
    learner_started = False
    try:
        for actor in actors:
            actor.start()
            started_actors.append(actor)
        learner.start()
        learner_started = True

        while True:
            alive_actors = [a for a in started_actors if a.is_alive()]
            if not alive_actors:
                break
            for actor in alive_actors:
                actor.join(timeout=0.2)
            if stop_requested['value']:
                raise KeyboardInterrupt

        learner.stop()
        while learner.is_alive():
            learner.join(timeout=1)
            if stop_requested['value']:
                break
    except KeyboardInterrupt:
        print('KeyboardInterrupt, force stopping workers...')
    finally:
        for actor in started_actors:
            actor.stop()

        alive_actors = _wait_processes(started_actors, timeout=5)
        for actor in alive_actors:
            actor.terminate()
        alive_actors = _wait_processes(alive_actors, timeout=5)
        for actor in alive_actors:
            print('%s did not exit after terminate(), sending kill()' % actor.name)
            actor.kill()
        _wait_processes(alive_actors, timeout=3)

        for actor in started_actors:
            try:
                actor.close()
            except Exception:
                pass

        if learner_started:
            if learner.is_alive():
                learner.stop()
            alive_learner = _wait_processes([learner], timeout=30)
            if alive_learner:
                learner.terminate()
                alive_learner = _wait_processes([learner], timeout=5)
            if alive_learner:
                print('Learner did not exit after terminate(), sending kill()')
                learner.kill()
                _wait_processes([learner], timeout=3)
            try:
                learner.close()
            except Exception:
                pass

        replay_buffer.close()

'''evaluate.py �?offline evaluation of a trained RL model.

Usage examples:

    # Evaluate RL model vs itself (self-play, measures consistency)
    python evaluate.py --model_b checkpoint/model_5000.pt --games 200

    # Evaluate RL model (B) vs SL baseline (A) �?A plays as the other 3 players
    python evaluate.py --model_b checkpoint/model_5000.pt --model_a /data/sl_model.pt --games 200

    # Use SL model architecture for baseline (loads only policy head from SL checkpoint)
    python evaluate.py --model_b checkpoint/model_5000.pt --model_a /data/sl_model.pt --model_a_is_sl --games 500
'''

import argparse
import numpy as np
import torch
import sys
import os

# Allow running from the RL directory directly
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model import CNNModel
from env import MahjongGBEnv
from feature import FeatureAgent


def _load_model(path, is_sl=False, device='cpu'):
    '''Load a CNNModel from a checkpoint path.

    is_sl=True  -> use load_sl_checkpoint() which tolerates missing _value_head.
    is_sl=False -> strict load; auto-fallback if _value_head keys are missing.
    '''
    model = CNNModel()
    if is_sl:
        model.load_sl_checkpoint(path, device=device)
    else:
        ckpt = torch.load(path, map_location=device)
        state_dict = ckpt.get('model', ckpt) if isinstance(ckpt, dict) and 'model' in ckpt else ckpt
        try:
            model.load_state_dict(state_dict)
        except RuntimeError as e:
            if '_value_head' in str(e):
                print('[evaluate] Warning: checkpoint missing _value_head (SL format). '
                      'Auto-switching to load_sl_checkpoint(). '
                      'Pass --model_b_is_sl to silence this.')
                model.load_sl_checkpoint(path, device=device)
            else:
                raise
    model.eval()
    return model


def _select_action(model, obs, greedy=True):
    obs_t  = torch.from_numpy(np.expand_dims(obs['observation'], 0))
    mask_t = torch.from_numpy(np.expand_dims(obs['action_mask'],  0))
    with torch.no_grad():
        logits, _ = model({'observation': obs_t, 'action_mask': mask_t})
    if greedy:
        return int(logits.argmax(dim=-1).item())
    else:
        return int(torch.distributions.Categorical(logits=logits).sample().item())


def evaluate(model_b, model_a, n_games=200, challenger_seat=0, greedy=True, verbose=False):
    '''Run n_games episodes; model_b plays at challenger_seat, model_a at all others.

    Returns a dict with:
        win_rate    �?fraction of games where model_b wins (Hu)
        avg_reward  �?average terminal reward for model_b
        avg_rank    �?average rank (1=best) for model_b based on terminal reward
        wins        �?absolute number of wins
    '''
    env_config = {
        'agent_clz'      : FeatureAgent,
        'duplicate'      : True,
        'reward_tenpai'  : 0.0,   # disable tenpai check during eval (very slow, ~136 fan-calc calls/game)
        'reward_notenpai': 0.0,
    }
    env = MahjongGBEnv(config=env_config)
    agent_names = env.agent_names
    challenger_name = agent_names[challenger_seat]

    wins         = 0
    total_reward = 0.0
    total_rank   = 0

    import time as _time
    _t_start = _time.time()

    MAX_STEPS = 500  # safety limit: a real game has at most ~136 draws

    for game in range(n_games):
        if verbose:
            print('Game %d/%d ...' % (game + 1, n_games), end=' ', flush=True)
        obs  = env.reset()
        done = False
        terminal_rewards = {name: 0.0 for name in agent_names}
        steps = 0

        while not done:
            actions = {}
            for name in obs:
                seat = agent_names.index(name)
                mdl  = model_b if seat == challenger_seat else model_a
                actions[name] = _select_action(mdl, obs[name], greedy=greedy)
            obs, rewards, done = env.step(actions)
            for name in rewards:
                terminal_rewards[name] += rewards[name]
            steps += 1
            if steps >= MAX_STEPS:
                if verbose:
                    print('[WARN] game %d hit MAX_STEPS=%d, forcing done' % (game+1, MAX_STEPS), flush=True)
                break

        # Compute rank: sort all players by reward descending, rank 1 = highest reward
        sorted_rewards = sorted(terminal_rewards.values(), reverse=True)
        challenger_reward = terminal_rewards[challenger_name]
        rank = sorted_rewards.index(challenger_reward) + 1   # 1-indexed

        total_reward += challenger_reward
        total_rank   += rank
        if challenger_reward > 0 and challenger_reward == max(terminal_rewards.values()):
            wins += 1

        if verbose:
            elapsed = _time.time() - _t_start
            speed = (game + 1) / elapsed if elapsed > 0 else 0
            eta = (n_games - game - 1) / speed if speed > 0 else float('inf')
            print('reward=%.1f rank=%d wins=%d(%.0f%%) %.1f g/min ETA %.0fs' % (
                challenger_reward, rank,
                wins, wins / (game + 1) * 100,
                speed * 60, eta,
            ), flush=True)

    return {
        'wins'       : wins,
        'win_rate'   : wins / n_games,
        'avg_reward' : total_reward / n_games,
        'avg_rank'   : total_rank   / n_games,
    }


def main():
    parser = argparse.ArgumentParser(description='Evaluate a trained Mahjong RL model.')
    parser.add_argument('--model_b', required=True,
                        help='Path to the challenger model checkpoint (.pt)')
    parser.add_argument('--model_a', default='',
                        help='Path to the baseline model checkpoint. '
                             'If omitted, model_b plays against itself.')
    parser.add_argument('--model_a_is_sl', action='store_true',
                        help='Load model_a as an SL checkpoint (remap _head �?_policy_head)')
    parser.add_argument('--model_b_is_sl', action='store_true',
                        help='Load model_b as an SL checkpoint (no value head)')
    parser.add_argument('--games', type=int, default=200,
                        help='Number of evaluation games (default: 200)')
    parser.add_argument('--challenger_seat', type=int, default=0,
                        help='Seat index [0-3] for the challenger model (default: 0)')
    parser.add_argument('--stochastic', action='store_true',
                        help='Use stochastic sampling instead of greedy action selection')
    parser.add_argument('--verbose', action='store_true',
                        help='Print progress every 50 games')
    args = parser.parse_args()

    print('Loading challenger model from %s...' % args.model_b)
    model_b = _load_model(args.model_b, is_sl=args.model_b_is_sl)

    if args.model_a:
        print('Loading baseline model from %s (is_sl=%s)...' % (args.model_a, args.model_a_is_sl))
        model_a = _load_model(args.model_a, is_sl=args.model_a_is_sl)
    else:
        print('No baseline specified �?challenger plays against itself.')
        model_a = model_b

    print('Running %d evaluation games (challenger at seat %d, greedy=%s)...' % (
        args.games, args.challenger_seat, not args.stochastic))

    results = evaluate(
        model_b          = model_b,
        model_a          = model_a,
        n_games          = args.games,
        challenger_seat  = args.challenger_seat,
        greedy           = not args.stochastic,
        verbose          = args.verbose,
    )

    print('\n── Evaluation Results ─────────────────────────')
    print('  Games      : %d' % args.games)
    print('  Wins       : %d' % results['wins'])
    print('  Win rate   : %.1f%%' % (results['win_rate'] * 100))
    print('  Avg reward : %.2f' % results['avg_reward'])
    print('  Avg rank   : %.2f' % results['avg_rank'])
    print('───────────────────────────────────────────────')


if __name__ == '__main__':
    main()

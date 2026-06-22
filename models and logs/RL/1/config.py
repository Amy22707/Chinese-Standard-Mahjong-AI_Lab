'''Centralised default hyperparameter configuration for RL training.

All values here act as defaults.  train.py argparse overrides them.
Import this module to get a fully-populated config dict:

    from config import DEFAULT_CONFIG
    config = dict(DEFAULT_CONFIG)
    config['lr'] = 3e-4  # override individual entries as needed
'''

import os

_BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DEFAULT_CONFIG = {
    # ── Infrastructure ────────────────────────────────────────────────────────
    'replay_buffer_size'    : 50000,   # max transitions in replay buffer
    'replay_buffer_episode' : 400,     # max episodes queued from actors
    'model_pool_size'       : 20,      # number of model snapshots kept in shared memory
    'model_pool_name'       : 'model-pool',

    # ── Actors ────────────────────────────────────────────────────────────────
    'num_actors'            : 24,      # parallel data-collection processes
    'episodes_per_actor'    : 1000,    # episodes each actor generates before stopping

    # ── PPO / GAE hyperparameters ─────────────────────────────────────────────
    'gamma'                 : 0.98,    # reward discount factor
    'lambda'                : 0.95,    # GAE λ – bias-variance trade-off
    'clip'                  : 0.2,     # PPO ε – trust-region clip for policy ratio
    'value_clip_range'      : 0.0,     # value-loss clip range; 0 = plain MSE (recommended).
                                       # If enabled, set to a value ≥ max_reward (e.g. 50.0)
                                       # NOT the policy clip_range (0.2 is far too small here!)
    'epochs'                : 5,       # PPO update epochs per data batch
    'batch_size'            : 256,     # minibatch size
    'min_sample'            : 200,     # minimum buffer size before training starts
    'value_coeff'           : 0.5,     # weight of value loss (lower weight reduces value noise)
    'entropy_coeff'         : 0.01,    # weight of entropy bonus (encourages exploration)
    'kl_coeff_start'        : 0.5,     # initial KL penalty to keep policy close to frozen SL teacher
    'kl_coeff_end'          : 0.01,    # final KL penalty after annealing
    'kl_anneal_updates'     : 100000,  # updates used for linear KL coefficient annealing
    'max_grad_norm'         : 0.5,     # gradient clipping threshold

    # ── Optimiser ─────────────────────────────────────────────────────────────
    'lr'                    : 3e-5,    # initial learning rate; lower is safer for SL warm-start
    'total_updates'         : 100000,  # total learner steps (used by LR scheduler)

    # ── Device ────────────────────────────────────────────────────────────────
    'device'                : 'auto',  # 'auto' | 'cpu' | 'cuda' | 'npu'

    # ── Checkpoints ───────────────────────────────────────────────────────────
    'ckpt_save_interval'    : 300,     # seconds between checkpoint saves
    'ckpt_save_path'        : os.path.join(_BASE_DIR, 'checkpoint') + os.sep,

    # ── Warm-start / resume ───────────────────────────────────────────────────
    'sl_checkpoint'         : '',      # path to SL .pt file for policy warm-start
    'resume_checkpoint'     : '',      # path to RL .pt checkpoint for resume

    # ── Logging ───────────────────────────────────────────────────────────────
    'tensorboard_dir'       : '',      # TensorBoard log dir; '' = disabled

    # ── Reward shaping ────────────────────────────────────────────────────────
    #   reward_scale: divide win/loss rewards by this factor for training stability
    #     recommend 10.0 when fan counts are large; 1.0 preserves original scale
    'reward_scale'          : 10.0,    # normalise win/loss to ~[-15, +30]; avoids value scale mismatch
    'reward_gang'           : 0.5,     # per-step bonus for completing a kong
    'reward_peng'           : 0.2,     # per-step bonus for completing a pong
    'reward_chi'            : 0.1,     # per-step bonus for completing a chow
    'reward_tenpai'         : 2.0,     # tenpai bonus at Huang (draw game)
    'reward_notenpai'       : -2.0,    # penalty for not being in tenpai at Huang
}

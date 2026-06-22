import torch
from torch import nn

try:
    from feature import FeatureAgent as _FA
    _DEFAULT_IN_CHANNELS = _FA.OBS_SIZE  # kept in sync with FeatureAgent automatically
except ImportError:
    _DEFAULT_IN_CHANNELS = 66            # fallback if feature.py is not importable here


def _build_action_types():
    '''Map each of the 235 action indices to one of 8 action-type buckets.
    Used by the decomposed policy head to build type-level supervision targets.
    Buckets: 0=Pass, 1=Hu, 2=Play, 3=Chi, 4=Peng, 5=Gang, 6=AnGang, 7=BuGang
    '''
    types = [0, 1]           # Pass, Hu
    types += [2] * 34        # Play (34 tiles)
    types += [3] * 63        # Chi  (3 suits * 7 mid tiles * 3 positions)
    types += [4] * 34        # Peng (34 tiles)
    types += [5] * 34        # Gang (34 tiles)
    types += [6] * 34        # AnGang (34 tiles)
    types += [7] * 34        # BuGang (34 tiles)
    return torch.LongTensor(types)


class ResidualBlock(nn.Module):
    '''ResBlock: Conv-BN-ReLU-Conv-BN + skip connection, then ReLU.'''

    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(True),
            nn.Conv2d(channels, channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.relu = nn.ReLU(True)

    def forward(self, x):
        return self.relu(x + self.layers(x))


class CNNModel(nn.Module):
    '''ResNet Actor-Critic for Chinese Standard Mahjong.

    Policy head mirrors the SL model exactly so that SL weights transfer cleanly:
        Shared trunk:  Conv2d(in_ch→128) → BN → ReLU → N × ResidualBlock
        Shared base:   Conv1×1-64 → BN → ReLU → Flatten → FC256 → ReLU → Dropout
        Action logits: type_head(FC8) + per-type sub-heads, assembled into 235-dim logits
        Value head:    Conv1×1-32 → BN → ReLU → Flatten → FC256 → ReLU → FC1

    The policy logit assembly:
        logits[0]      = type[0]           (Pass)
        logits[1]      = type[1]           (Hu)
        logits[2:36]   = type[2] + play_head(x)    (Play, 34 tiles)
        logits[36:99]  = type[3] + chi_head(x)     (Chi, 63 combos)
        logits[99:133] = type[4] + peng_head(x)    (Peng, 34 tiles)
        ... etc.
    This decomposition makes the model parameter-efficient and matches the SL training target.
    '''

    def __init__(self, in_channels=_DEFAULT_IN_CHANNELS, hidden_channels=128, blocks=6):
        super(CNNModel, self).__init__()

        # Shared trunk (convolutional feature extraction)
        self._trunk = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(True),
            *[ResidualBlock(hidden_channels) for _ in range(blocks)]
        )

        # Shared feature base before both policy and (optionally) value heads
        self._head_base = nn.Sequential(
            nn.Conv2d(hidden_channels, 64, 1, 1, 0, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.Flatten(),
            nn.Linear(64 * 4 * 9, 256),
            nn.ReLU(True),
            nn.Dropout(0.1),
        )

        # Decomposed policy head: one type head + per-action-type sub-heads
        self._type_head   = nn.Linear(256, 8)
        self._play_head   = nn.Linear(256, 34)
        self._chi_head    = nn.Linear(256, 63)
        self._peng_head   = nn.Linear(256, 34)
        self._gang_head   = nn.Linear(256, 34)
        self._angang_head = nn.Linear(256, 34)
        self._bugang_head = nn.Linear(256, 34)

        # Register action-type mapping as a non-persistent buffer (no grad, saved with model)
        self.register_buffer('_action_types', _build_action_types(), persistent=False)

        # Value head (critic) – independent from policy head
        self._value_head = nn.Sequential(
            nn.Conv2d(hidden_channels, 32, 1, 1, 0, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.Flatten(),
            nn.Linear(32 * 4 * 9, 256),
            nn.ReLU(True),
            nn.Linear(256, 1),
        )

        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, input_dict):
        obs = input_dict['observation'].float()
        trunk_out = self._trunk(obs)

        # ── Policy head (decomposed) ───────────────────────────────────────
        base = self._head_base(trunk_out)
        type_logits = self._type_head(base)
        # Assemble 235-dim logit vector from type + sub-head contributions
        action_logits = obs.new_empty((obs.shape[0], 235))
        action_logits[:, 0]        = type_logits[:, 0]
        action_logits[:, 1]        = type_logits[:, 1]
        action_logits[:, 2:36]     = type_logits[:, 2:3] + self._play_head(base)
        action_logits[:, 36:99]    = type_logits[:, 3:4] + self._chi_head(base)
        action_logits[:, 99:133]   = type_logits[:, 4:5] + self._peng_head(base)
        action_logits[:, 133:167]  = type_logits[:, 5:6] + self._gang_head(base)
        action_logits[:, 167:201]  = type_logits[:, 6:7] + self._angang_head(base)
        action_logits[:, 201:235]  = type_logits[:, 7:8] + self._bugang_head(base)
        # Mask illegal actions with a large negative value
        action_mask = input_dict['action_mask'].bool()
        masked_logits = action_logits.masked_fill(~action_mask, -1e9)

        # ── Value head (critic) ────────────────────────────────────────────
        value = self._value_head(trunk_out)

        return masked_logits, value

    def action_type_logits(self, input_dict):
        '''Return raw 8-dim action-type logits (used by SL auxiliary loss).'''
        obs = input_dict['observation'].float()
        return self._type_head(self._head_base(self._trunk(obs)))

    def action_type_targets(self, actions):
        '''Map flat action indices to action-type bucket indices (0-7).'''
        return self._action_types.to(actions.device)[actions]

    def load_sl_checkpoint(self, path, device='cpu'):
        '''Warm-start trunk + policy head from an SL model checkpoint.

        Both the SL and RL models share identical _trunk, _head_base, _type_head,
        and all per-action sub-head names, so all policy weights transfer directly.
        Only the value head (RL-only) keeps its random initialisation.
        '''
        raw = torch.load(path, map_location=device)
        # Support {'model': state_dict, ...} format (RL checkpoint) or raw state_dict (SL)
        sl_state = raw.get('model', raw) if isinstance(raw, dict) and 'model' in raw else raw
        own_state = self.state_dict()
        loaded, skipped = 0, 0
        for name, param in sl_state.items():
            if name in own_state and own_state[name].shape == param.shape:
                own_state[name].copy_(param)
                loaded += 1
            else:
                skipped += 1
        self.load_state_dict(own_state)
        print('[CNNModel] Loaded %d params from SL checkpoint %s (skipped %d).' % (loaded, path, skipped))

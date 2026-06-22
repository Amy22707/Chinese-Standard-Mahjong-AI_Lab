import torch
from torch import nn

try:
    from feature import FeatureAgent as _FA
    _DEFAULT_IN_CHANNELS = _FA.OBS_SIZE   # kept in sync with FeatureAgent automatically
except ImportError:
    _DEFAULT_IN_CHANNELS = 70             # fallback if feature.py is not importable here


def _build_action_types():
    action_types = [0, 1]
    action_types += [2] * 34
    action_types += [3] * 63
    action_types += [4] * 34
    action_types += [5] * 34
    action_types += [6] * 34
    action_types += [7] * 34
    return torch.LongTensor(action_types)


class ResidualBlock(nn.Module):

    def __init__(self, channels):
        nn.Module.__init__(self)
        self.layers = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1, bias = False),
            nn.BatchNorm2d(channels),
            nn.ReLU(True),
            nn.Conv2d(channels, channels, 3, 1, 1, bias = False),
            nn.BatchNorm2d(channels)
        )
        self.relu = nn.ReLU(True)

    def forward(self, x):
        return self.relu(x + self.layers(x))


class CNNModel(nn.Module):

    def __init__(self, in_channels = _DEFAULT_IN_CHANNELS, hidden_channels = 128, blocks = 6,
                 seq_embed_dim = 16, seq_hidden_dim = 64):
        nn.Module.__init__(self)
        self._trunk = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, 3, 1, 1, bias = False),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(True),
            *(ResidualBlock(hidden_channels) for _ in range(blocks))
        )
        self._head_base = nn.Sequential(
            nn.Conv2d(hidden_channels, 64, 1, 1, 0, bias = False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.Flatten(),
            nn.Linear(64 * 4 * 9, 256),
            nn.ReLU(True),
            nn.Dropout(0.1)
        )
        self._tile_embed = nn.Embedding(35, seq_embed_dim, padding_idx = 34)
        self._player_embed = nn.Embedding(4, 4)
        self._seq_gru = nn.GRU(seq_embed_dim + 4, seq_hidden_dim, batch_first = True)
        feature_dim = 256 + seq_hidden_dim
        self._fusion = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(True),
            nn.Dropout(0.1)
        )
        self._type_head = nn.Linear(256, 8)
        self._play_head = nn.Linear(256, 34)
        self._chi_head = nn.Linear(256, 63)
        self._peng_head = nn.Linear(256, 34)
        self._gang_head = nn.Linear(256, 34)
        self._angang_head = nn.Linear(256, 34)
        self._bugang_head = nn.Linear(256, 34)
        self._win_head = nn.Linear(256, 1)
        self._fan_head = nn.Linear(256, 1)
        self._shanten_head = nn.Linear(256, 1)
        self._discard_rank_head = nn.Linear(256, 34)
        self._risk_head = nn.Linear(256, 34)
        self._fan_route_head = nn.Linear(256, 5)
        self.register_buffer('_action_types', _build_action_types(), persistent = False)
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _encode(self, input_dict):
        obs = input_dict["observation"].float()
        board_x = self._head_base(self._trunk(obs))
        seq_tile = input_dict.get('discard_seq')
        seq_player = input_dict.get('discard_player')
        if seq_tile is None or seq_player is None:
            seq_x = board_x.new_zeros((board_x.shape[0], self._seq_gru.hidden_size))
        else:
            seq_tile = seq_tile.long().clamp(0, 34)
            seq_player = seq_player.long().clamp(0, 3)
            seq_emb = torch.cat([self._tile_embed(seq_tile), self._player_embed(seq_player)], dim = -1)
            _, h = self._seq_gru(seq_emb)
            seq_x = h[-1]
        return self._fusion(torch.cat([board_x, seq_x], dim = 1))

    def forward(self, input_dict, return_type_logits = False, return_aux = False):
        obs = input_dict["observation"].float()
        x = self._encode(input_dict)
        type_logits = self._type_head(x)
        action_logits = obs.new_empty((obs.shape[0], 235))
        action_logits[:, 0] = type_logits[:, 0]
        action_logits[:, 1] = type_logits[:, 1]
        action_logits[:, 2 : 36] = type_logits[:, 2:3] + self._play_head(x)
        action_logits[:, 36 : 99] = type_logits[:, 3:4] + self._chi_head(x)
        action_logits[:, 99 : 133] = type_logits[:, 4:5] + self._peng_head(x)
        action_logits[:, 133 : 167] = type_logits[:, 5:6] + self._gang_head(x)
        action_logits[:, 167 : 201] = type_logits[:, 6:7] + self._angang_head(x)
        action_logits[:, 201 : 235] = type_logits[:, 7:8] + self._bugang_head(x)
        action_mask = input_dict["action_mask"].bool()
        action_logits = action_logits.masked_fill(~action_mask, -100.0)
        aux = None
        if return_aux:
            aux = {
                'win_logit'     : self._win_head(x).squeeze(-1),
                'fan'           : self._fan_head(x).squeeze(-1),
                'shanten'       : self._shanten_head(x).squeeze(-1),
                'discard_rank'  : self._discard_rank_head(x),
                'risk'          : self._risk_head(x),
                'fan_route'     : self._fan_route_head(x),
            }
        if return_type_logits:
            if return_aux:
                return action_logits, type_logits, aux
            return action_logits, type_logits
        if return_aux:
            return action_logits, aux
        return action_logits

    def action_type_logits(self, input_dict):
        return self._type_head(self._encode(input_dict))

    def action_type_targets(self, actions):
        return self._action_types.to(actions.device)[actions]

    def subaction_logits(self, input_dict):
        x = self._encode(input_dict)
        return {
            2: self._play_head(x),
            3: self._chi_head(x),
            4: self._peng_head(x),
            5: self._gang_head(x),
            6: self._angang_head(x),
            7: self._bugang_head(x),
        }

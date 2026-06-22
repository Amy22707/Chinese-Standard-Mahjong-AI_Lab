import torch
from torch import nn

try:
    from feature import FeatureAgent as _FA
    _DEFAULT_IN_CHANNELS = _FA.OBS_SIZE   # kept in sync with FeatureAgent automatically
except ImportError:
    _DEFAULT_IN_CHANNELS = 51             # fallback if feature.py is not importable here


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

    def __init__(self, in_channels = _DEFAULT_IN_CHANNELS, hidden_channels = 128, blocks = 6):
        nn.Module.__init__(self)
        self._trunk = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, 3, 1, 1, bias = False),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(True),
            *(ResidualBlock(hidden_channels) for _ in range(blocks))
        )
        self._head = nn.Sequential(
            nn.Conv2d(hidden_channels, 64, 1, 1, 0, bias = False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.Flatten(),
            nn.Linear(64 * 4 * 9, 256),
            nn.ReLU(True),
            nn.Dropout(0.1),
            nn.Linear(256, 235)
        )
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, input_dict):
        obs = input_dict["observation"].float()
        action_logits = self._head(self._trunk(obs))
        action_mask = input_dict["action_mask"].bool()
        return action_logits.masked_fill(~action_mask, -100.0)

from torch.utils.data import Dataset
import numpy as np
from bisect import bisect_right
from collections import OrderedDict


SUIT_PERMUTATIONS = [
    (0, 1, 2),
    (0, 2, 1),
    (1, 0, 2),
    (1, 2, 0),
    (2, 0, 1),
    (2, 1, 0)
]


def build_tile_permutation(suit_perm):
    tile_perm = np.arange(36)
    for old_suit, new_suit in enumerate(suit_perm):
        for num in range(9):
            tile_perm[old_suit * 9 + num] = new_suit * 9 + num
    return tile_perm


def build_action_permutation(suit_perm):
    tile_perm = build_tile_permutation(suit_perm)
    action_perm = np.arange(235)
    action_perm[2 : 36] = 2 + tile_perm[:34]
    for offset in (99, 133, 167, 201):
        action_perm[offset : offset + 34] = offset + tile_perm[:34]
    for old_suit, new_suit in enumerate(suit_perm):
        old_begin = 36 + old_suit * 21
        new_begin = 36 + new_suit * 21
        action_perm[old_begin : old_begin + 21] = np.arange(new_begin, new_begin + 21)
    return tile_perm, action_perm


PERMUTATIONS = [build_action_permutation(p) for p in SUIT_PERMUTATIONS]


def build_discard_rank_target(mask, act):
    target = np.full(34, -1.0, dtype = np.float32)
    play_begin = 2
    play_end = 36
    legal = np.nonzero(mask[play_begin : play_end])[0]
    if len(legal):
        target[legal] = 0.5
    if play_begin <= act < play_end:
        target[int(act - play_begin)] = 1.0
    return target


def infer_fan_route(obs):
    '''0 balanced, 1 chiitoi, 2 flush, 3 triplet, 4 honour-heavy.'''
    try:
        hand = obs[2:6].sum(axis = 0).reshape(36)
        hand = (hand > 0).astype(np.float32)
        chiitoi = float(obs[62, 0, 0]) if obs.shape[0] > 62 else 1.0
        flush = obs[63:66, 0, 0] if obs.shape[0] > 65 else np.ones(3, dtype = np.float32)
        honours = hand[27:34].sum()
        if chiitoi <= 2.0 / 6.0:
            return 1
        if float(np.min(flush)) <= 4.0 / 13.0:
            return 2
        if honours >= 4:
            return 4
        return 0
    except Exception:
        return 0


class _LRUFileCache:
    '''Thread-unsafe LRU cache for .npz match files (one instance per DataLoader worker).'''

    def __init__(self, maxsize):
        self._cache = OrderedDict()
        self._maxsize = maxsize

    def get(self, path):
        if path in self._cache:
            self._cache.move_to_end(path)
            return self._cache[path]
        # Load and materialise arrays so the NpzFile handle can be closed immediately
        with np.load(path) as npz:
            data = {k: npz[k] for k in npz}
        self._cache[path] = data
        if len(self._cache) > self._maxsize:
            self._cache.popitem(last = False)
        return data


class MahjongGBDataset(Dataset):
    
    def __init__(self, begin = 0, end = 1, augment = False, cache_size = 256):
        import json
        with open('data/count.json') as f:
            self.match_samples = json.load(f)
        self.total_matches = len(self.match_samples)
        self.total_samples = sum(self.match_samples)
        self.begin = int(begin * self.total_matches)
        self.end = int(end * self.total_matches)
        self.match_samples = self.match_samples[self.begin : self.end]
        self.matches = len(self.match_samples)
        self.samples = sum(self.match_samples)
        self.augment = augment
        # Convert per-match sample counts to cumulative offsets for bisect lookup
        t = 0
        for i in range(self.matches):
            a = self.match_samples[i]
            self.match_samples[i] = t
            t += a
        # Lazy file cache; populated on first access per worker
        self._cache_size = cache_size
        self._file_cache = _LRUFileCache(cache_size)
    
    def __len__(self):
        return self.samples
    
    def _load_match(self, match_id):
        path = 'data/%d.npz' % (match_id + self.begin)
        return self._file_cache.get(path)

    def sample_weights(self):
        weights = []
        for match_id in range(self.matches):
            data = self._load_match(match_id)
            if 'wt' in data:
                weights.extend(data['wt'].astype(np.float32).tolist())
            else:
                weights.extend([1.0] * len(data['act']))
        return np.asarray(weights, dtype = np.float32)

    def __getitem__(self, index):
        match_id = bisect_right(self.match_samples, index, 0, self.matches) - 1
        sample_id = index - self.match_samples[match_id]
        data = self._load_match(match_id)
        obs  = data['obs'][sample_id]
        mask = data['mask'][sample_id]
        act  = data['act'][sample_id]
        if obs.shape[0] < 70:
            obs = np.pad(obs, ((0, 70 - obs.shape[0]), (0, 0), (0, 0)), mode = 'constant')
        wt   = float(data['wt'][sample_id]) if 'wt' in data else 1.0
        win = float(data['win'][sample_id]) if 'win' in data else 1.0
        fan = float(data['fan'][sample_id]) if 'fan' in data else max(0.0, min(1.0, (wt - 1.0) / 4.0))
        shanten = float(data['shanten'][sample_id]) if 'shanten' in data else float(obs[60, 0, 0])
        discard_rank = build_discard_rank_target(mask, act)
        risk = data['risk'][sample_id].astype(np.float32) if 'risk' in data else np.zeros(34, dtype = np.float32)
        fan_route = int(data['fan_route'][sample_id]) if 'fan_route' in data else infer_fan_route(obs)
        discard_seq = data['seq_tile'][sample_id].astype(np.int64) if 'seq_tile' in data else np.full(80, 34, dtype = np.int64)
        discard_player = data['seq_player'][sample_id].astype(np.int64) if 'seq_player' in data else np.zeros(80, dtype = np.int64)
        if not self.augment:
            return obs, mask, act, wt, discard_seq, discard_player, {
                'win': win,
                'fan': fan,
                'shanten': shanten,
                'discard_rank': discard_rank,
                'risk': risk,
                'fan_route': fan_route,
            }
        tile_perm, action_perm = PERMUTATIONS[np.random.randint(len(PERMUTATIONS))]
        aug_obs = obs.reshape(obs.shape[0], 36)[:, tile_perm].reshape(obs.shape).copy()
        aug_mask = np.zeros_like(mask)
        aug_mask[action_perm[np.nonzero(mask)[0]]] = 1
        aug_act = action_perm[act]
        aug_risk = np.zeros_like(risk)
        aug_risk[tile_perm[:34]] = risk
        aug_seq = discard_seq.copy()
        valid_seq = aug_seq < 34
        aug_seq[valid_seq] = tile_perm[aug_seq[valid_seq]]
        return aug_obs, aug_mask, aug_act, wt, aug_seq, discard_player, {
            'win': win,
            'fan': fan,
            'shanten': shanten,
            'discard_rank': build_discard_rank_target(aug_mask, aug_act),
            'risk': aug_risk,
            'fan_route': infer_fan_route(aug_obs) if fan_route in (1, 2, 4) else fan_route,
        }

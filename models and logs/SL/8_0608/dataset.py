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

    def __getitem__(self, index):
        match_id = bisect_right(self.match_samples, index, 0, self.matches) - 1
        sample_id = index - self.match_samples[match_id]
        data = self._load_match(match_id)
        obs  = data['obs'][sample_id]
        mask = data['mask'][sample_id]
        act  = data['act'][sample_id]
        wt   = float(data['wt'][sample_id]) if 'wt' in data else 1.0
        if not self.augment:
            return obs, mask, act, wt
        tile_perm, action_perm = PERMUTATIONS[np.random.randint(len(PERMUTATIONS))]
        aug_obs = obs.reshape(obs.shape[0], 36)[:, tile_perm].reshape(obs.shape).copy()
        aug_mask = np.zeros_like(mask)
        aug_mask[action_perm[np.nonzero(mask)[0]]] = 1
        return aug_obs, aug_mask, action_perm[act], wt

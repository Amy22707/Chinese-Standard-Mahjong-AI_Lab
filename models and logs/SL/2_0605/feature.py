from agent import MahjongGBAgent
from collections import defaultdict
import numpy as np

try:
    from MahjongGB import MahjongFanCalculator
except:
    print('MahjongGB library required! Please visit https://github.com/ailab-pku/PyMahjongGB for more information.')
    raise

class FeatureAgent(MahjongGBAgent):
    
    '''
    observation: 60*4*9
        seat wind(1) + prevalent wind(1) + hand(4) + shown(4) + remaining(4) +
        last discard(1) + wall(4) + discard_p0..p3(4*4) + pack_p0..p3(4*4) +
        about_kong(1) + wall_last(1) + danger_opp1..3(3) + discard_pos_p0..p3(4)
        = 51 + 1 + 1 + 3 + 4 = 60 channels, each over 4*9 tile grid
    action_mask: 235
        pass1+hu1+discard34+chi63(3*7*3)+peng34+gang34+angang34+bugang34
    '''
    
    OBS_SIZE = 60
    ACT_SIZE = 235
    
    OFFSET_OBS = {
        'SEAT_WIND'      : 0,   # 1 ch: one-hot seat wind tile
        'PREVALENT_WIND' : 1,   # 1 ch: one-hot prevalent wind tile
        'HAND'           : 2,   # 4 ch: my hand count encoding (ch k = has ≥k+1 copies)
        'SHOWN'          : 6,   # 4 ch: all visible tiles (discard + melds)
        'REMAINING'      : 10,  # 4 ch: tiles not in my hand or visible
        'LAST_DISCARD'   : 14,  # 1 ch: the most recent discard
        'WALL'           : 15,  # 4 ch: tiles remaining per player wall (normalised)
        'DISCARD'        : 19,  # 16 ch: per-player discard history (4 ch each, p0..p3)
        'PACK'           : 35,  # 16 ch: per-player meld tiles (4 ch each, p0..p3)
        'ABOUT_KONG'     : 51,  # 1 ch: scalar flag broadcast – current draw is after a kong
        'WALL_LAST'      : 52,  # 1 ch: scalar flag broadcast – wall is at last tile
        'DANGER'         : 53,  # 3 ch: opponent seats 1/2/3 – 1 where they have NOT discarded that tile
        'DISCARD_POS'    : 56,  # 4 ch: per player – normalised recency of each tile in discard seq
    }
    OFFSET_ACT = {
        'Pass' : 0,
        'Hu' : 1,
        'Play' : 2,
        'Chi' : 36,
        'Peng' : 99,
        'Gang' : 133,
        'AnGang' : 167,
        'BuGang' : 201
    }
    TILE_LIST = [
        *('W%d'%(i+1) for i in range(9)),
        *('T%d'%(i+1) for i in range(9)),
        *('B%d'%(i+1) for i in range(9)),
        *('F%d'%(i+1) for i in range(4)),
        *('J%d'%(i+1) for i in range(3))
    ]
    OFFSET_TILE = {c : i for i, c in enumerate(TILE_LIST)}
    
    def __init__(self, seatWind):
        self.seatWind = seatWind
        self.packs = [[] for i in range(4)]
        self.history = [[] for i in range(4)]
        self.tileWall = [21] * 4
        self.shownTiles = defaultdict(int)
        self.lastDiscard = None
        self.wallLast = False
        self.isAboutKong = False
        self.obs = np.zeros((self.OBS_SIZE, 36))
        self.obs[self.OFFSET_OBS['SEAT_WIND']][self.OFFSET_TILE['F%d' % (self.seatWind + 1)]] = 1
    
    '''
    Wind 0..3
    Deal XX XX ...
    Player N Draw
    Player N Gang
    Player N(me) AnGang XX
    Player N(me) Play XX
    Player N(me) BuGang XX
    Player N(not me) Peng
    Player N(not me) Chi XX
    Player N(not me) AnGang
    
    Player N Hu
    Huang
    Player N Invalid
    Draw XX
    Player N(not me) Play XX
    Player N(not me) BuGang XX
    Player N(me) Peng
    Player N(me) Chi XX
    '''
    def request2obs(self, request):
        t = request.split()
        if t[0] == 'Wind':
            self.prevalentWind = int(t[1])
            self.obs[self.OFFSET_OBS['PREVALENT_WIND']][self.OFFSET_TILE['F%d' % (self.prevalentWind + 1)]] = 1
            return
        if t[0] == 'Deal':
            self.hand = t[1:]
            self._hand_embedding_update()
            return
        if t[0] == 'Huang':
            self.valid = []
            return self._obs()
        if t[0] == 'Draw':
            # Available: Hu, Play, AnGang, BuGang
            self.tileWall[0] -= 1
            self.wallLast = self.tileWall[1] == 0
            tile = t[1]
            self.valid = []
            if self._check_mahjong(tile, isSelfDrawn = True, isAboutKong = self.isAboutKong):
                self.valid.append(self.OFFSET_ACT['Hu'])
            self.isAboutKong = False
            self.hand.append(tile)
            self._hand_embedding_update()
            for tile in set(self.hand):
                self.valid.append(self.OFFSET_ACT['Play'] + self.OFFSET_TILE[tile])
                if self.hand.count(tile) == 4 and not self.wallLast and self.tileWall[0] > 0:
                    self.valid.append(self.OFFSET_ACT['AnGang'] + self.OFFSET_TILE[tile])
            if not self.wallLast and self.tileWall[0] > 0:
                for packType, tile, offer in self.packs[0]:
                    if packType == 'PENG' and tile in self.hand:
                        self.valid.append(self.OFFSET_ACT['BuGang'] + self.OFFSET_TILE[tile])
            return self._obs()
        # Player N Invalid/Hu/Draw/Play/Chi/Peng/Gang/AnGang/BuGang XX
        p = (int(t[1]) + 4 - self.seatWind) % 4
        if t[2] == 'Draw':
            self.tileWall[p] -= 1
            self.wallLast = self.tileWall[(p + 1) % 4] == 0
            return
        if t[2] == 'Invalid':
            self.valid = []
            return self._obs()
        if t[2] == 'Hu':
            self.valid = []
            return self._obs()
        if t[2] == 'Play':
            self.tileFrom = p
            self.curTile = t[3]
            self.lastDiscard = self.curTile
            self.shownTiles[self.curTile] += 1
            self.history[p].append(self.curTile)
            if p == 0:
                self.hand.remove(self.curTile)
                self._hand_embedding_update()
                return
            else:
                # Available: Hu/Gang/Peng/Chi/Pass
                self.valid = []
                if self._check_mahjong(self.curTile):
                    self.valid.append(self.OFFSET_ACT['Hu'])
                if not self.wallLast:
                    if self.hand.count(self.curTile) >= 2:
                        self.valid.append(self.OFFSET_ACT['Peng'] + self.OFFSET_TILE[self.curTile])
                        if self.hand.count(self.curTile) == 3 and self.tileWall[0]:
                            self.valid.append(self.OFFSET_ACT['Gang'] + self.OFFSET_TILE[self.curTile])
                    color = self.curTile[0]
                    if p == 3 and color in 'WTB':
                        num = int(self.curTile[1])
                        tmp = []
                        for i in range(-2, 3): tmp.append(color + str(num + i))
                        if tmp[0] in self.hand and tmp[1] in self.hand:
                            self.valid.append(self.OFFSET_ACT['Chi'] + 'WTB'.index(color) * 21 + (num - 3) * 3 + 2)
                        if tmp[1] in self.hand and tmp[3] in self.hand:
                            self.valid.append(self.OFFSET_ACT['Chi'] + 'WTB'.index(color) * 21 + (num - 2) * 3 + 1)
                        if tmp[3] in self.hand and tmp[4] in self.hand:
                            self.valid.append(self.OFFSET_ACT['Chi'] + 'WTB'.index(color) * 21 + (num - 1) * 3)
                self.valid.append(self.OFFSET_ACT['Pass'])
                return self._obs()
        if t[2] == 'Chi':
            tile = t[3]
            color = tile[0]
            num = int(tile[1])
            self.packs[p].append(('CHI', tile, int(self.curTile[1]) - num + 2))
            self.shownTiles[self.curTile] -= 1
            for i in range(-1, 2):
                self.shownTiles[color + str(num + i)] += 1
            self.wallLast = self.tileWall[(p + 1) % 4] == 0
            if p == 0:
                # Available: Play
                self.valid = []
                self.hand.append(self.curTile)
                for i in range(-1, 2):
                    self.hand.remove(color + str(num + i))
                self._hand_embedding_update()
                for tile in set(self.hand):
                    self.valid.append(self.OFFSET_ACT['Play'] + self.OFFSET_TILE[tile])
                return self._obs()
            else:
                return
        if t[2] == 'UnChi':
            tile = t[3]
            color = tile[0]
            num = int(tile[1])
            self.packs[p].pop()
            self.shownTiles[self.curTile] += 1
            for i in range(-1, 2):
                self.shownTiles[color + str(num + i)] -= 1
            if p == 0:
                for i in range(-1, 2):
                    self.hand.append(color + str(num + i))
                self.hand.remove(self.curTile)
                self._hand_embedding_update()
            return
        if t[2] == 'Peng':
            self.packs[p].append(('PENG', self.curTile, (4 + p - self.tileFrom) % 4))
            self.shownTiles[self.curTile] += 2
            self.wallLast = self.tileWall[(p + 1) % 4] == 0
            if p == 0:
                # Available: Play
                self.valid = []
                for i in range(2):
                    self.hand.remove(self.curTile)
                self._hand_embedding_update()
                for tile in set(self.hand):
                    self.valid.append(self.OFFSET_ACT['Play'] + self.OFFSET_TILE[tile])
                return self._obs()
            else:
                return
        if t[2] == 'UnPeng':
            self.packs[p].pop()
            self.shownTiles[self.curTile] -= 2
            if p == 0:
                for i in range(2):
                    self.hand.append(self.curTile)
                self._hand_embedding_update()
            return
        if t[2] == 'Gang':
            self.packs[p].append(('GANG', self.curTile, (4 + p - self.tileFrom) % 4))
            self.shownTiles[self.curTile] += 3
            if p == 0:
                for i in range(3):
                    self.hand.remove(self.curTile)
                self._hand_embedding_update()
                self.isAboutKong = True
            return
        if t[2] == 'AnGang':
            tile = 'CONCEALED' if p else t[3]
            self.packs[p].append(('GANG', tile, 0))
            if p == 0:
                self.isAboutKong = True
                for i in range(4):
                    self.hand.remove(tile)
            else:
                self.isAboutKong = False
            return
        if t[2] == 'BuGang':
            tile = t[3]
            for i in range(len(self.packs[p])):
                if tile == self.packs[p][i][1]:
                    self.packs[p][i] = ('GANG', tile, self.packs[p][i][2])
                    break
            self.shownTiles[tile] += 1
            if p == 0:
                self.hand.remove(tile)
                self._hand_embedding_update()
                self.isAboutKong = True
                return
            else:
                # Available: Hu/Pass
                self.valid = []
                if self._check_mahjong(tile, isSelfDrawn = False, isAboutKong = True):
                    self.valid.append(self.OFFSET_ACT['Hu'])
                self.valid.append(self.OFFSET_ACT['Pass'])
            return self._obs()
        raise NotImplementedError('Unknown request %s!' % request)
    
    '''
    Pass
    Hu
    Play XX
    Chi XX
    Peng
    Gang
    (An)Gang XX
    BuGang XX
    '''
    def action2response(self, action):
        if action < self.OFFSET_ACT['Hu']:
            return 'Pass'
        if action < self.OFFSET_ACT['Play']:
            return 'Hu'
        if action < self.OFFSET_ACT['Chi']:
            return 'Play ' + self.TILE_LIST[action - self.OFFSET_ACT['Play']]
        if action < self.OFFSET_ACT['Peng']:
            t = (action - self.OFFSET_ACT['Chi']) // 3
            return 'Chi ' + 'WTB'[t // 7] + str(t % 7 + 2)
        if action < self.OFFSET_ACT['Gang']:
            return 'Peng'
        if action < self.OFFSET_ACT['AnGang']:
            return 'Gang'
        if action < self.OFFSET_ACT['BuGang']:
            return 'Gang ' + self.TILE_LIST[action - self.OFFSET_ACT['AnGang']]
        return 'BuGang ' + self.TILE_LIST[action - self.OFFSET_ACT['BuGang']]
    
    '''
    Pass
    Hu
    Play XX
    Chi XX
    Peng
    Gang
    (An)Gang XX
    BuGang XX
    '''
    def response2action(self, response):
        t = response.split()
        if t[0] == 'Pass': return self.OFFSET_ACT['Pass']
        if t[0] == 'Hu': return self.OFFSET_ACT['Hu']
        if t[0] == 'Play': return self.OFFSET_ACT['Play'] + self.OFFSET_TILE[t[1]]
        if t[0] == 'Chi': return self.OFFSET_ACT['Chi'] + 'WTB'.index(t[1][0]) * 7 * 3 + (int(t[2][1]) - 2) * 3 + int(t[1][1]) - int(t[2][1]) + 1
        if t[0] == 'Peng': return self.OFFSET_ACT['Peng'] + self.OFFSET_TILE[t[1]]
        if t[0] == 'Gang': return self.OFFSET_ACT['Gang'] + self.OFFSET_TILE[t[1]]
        if t[0] == 'AnGang': return self.OFFSET_ACT['AnGang'] + self.OFFSET_TILE[t[1]]
        if t[0] == 'BuGang': return self.OFFSET_ACT['BuGang'] + self.OFFSET_TILE[t[1]]
        return self.OFFSET_ACT['Pass']
    
    def _obs(self):
        mask = np.zeros(self.ACT_SIZE)
        for a in self.valid:
            mask[a] = 1
        self._public_embedding_update()
        return {
            'observation': self.obs.reshape((self.OBS_SIZE, 4, 9)).astype(np.float32).copy(),
            'action_mask': mask
        }
    
    def _hand_embedding_update(self):
        self.obs[self.OFFSET_OBS['HAND'] : self.OFFSET_OBS['HAND'] + 4] = 0
        d = defaultdict(int)
        for tile in self.hand:
            d[tile] += 1
        self._count_embedding(self.OFFSET_OBS['HAND'], d)
    
    def _public_embedding_update(self):
        # Reset all public channels (SHOWN through end of PACK block)
        self.obs[self.OFFSET_OBS['SHOWN'] : self.OBS_SIZE] = 0

        # All visible tiles (opponents' discards + all exposed melds)
        self._count_embedding(self.OFFSET_OBS['SHOWN'], self.shownTiles)

        # Remaining tiles (not in my hand and not yet visible)
        handCnt = defaultdict(int)
        for tile in self.hand:
            handCnt[tile] += 1
        remaining = defaultdict(int)
        for tile in self.TILE_LIST:
            remaining[tile] = max(0, 4 - handCnt[tile] - self.shownTiles[tile])
        self._count_embedding(self.OFFSET_OBS['REMAINING'], remaining)

        if self.lastDiscard is not None:
            self.obs[self.OFFSET_OBS['LAST_DISCARD'], self.OFFSET_TILE[self.lastDiscard]] = 1

        for i, wall in enumerate(self.tileWall):
            self.obs[self.OFFSET_OBS['WALL'] + i, :] = wall / 21

        # Per-player discard history (relative seat order: 0=self, 1=left, 2=opposite, 3=right)
        for p in range(4):
            offset = self.OFFSET_OBS['DISCARD'] + p * 4
            d = defaultdict(int)
            for tile in self.history[p]:
                d[tile] += 1
            self._count_embedding(offset, d)

        # Per-player meld tiles (count of tiles committed to packs)
        for p in range(4):
            offset = self.OFFSET_OBS['PACK'] + p * 4
            self._count_embedding(offset, self._get_pack_tile_counts(p))

        # ── 新增特征 ──────────────────────────────────────────────────────────

        # About-Kong flag: broadcast scalar over all 36 tile slots
        self.obs[self.OFFSET_OBS['ABOUT_KONG'], :] = float(self.isAboutKong)

        # Wall-last flag: broadcast scalar over all 36 tile slots
        self.obs[self.OFFSET_OBS['WALL_LAST'], :] = float(self.wallLast)

        # Danger estimate (one channel per opponent, relative seats 1/2/3):
        #   value = 1 if that opponent has NOT yet discarded the tile → tile may still be wanted
        #   value = 0 if the opponent has discarded it → relatively safer to play
        for k in range(3):
            ch = self.OFFSET_OBS['DANGER'] + k
            self.obs[ch, :] = 1  # assume all tiles potentially dangerous
            for tile in set(self.history[k + 1]):
                if tile in self.OFFSET_TILE:
                    self.obs[ch, self.OFFSET_TILE[tile]] = 0  # discarded = not held

        # Discard positional encoding (one channel per player):
        #   value = (most-recent discard index of tile + 1) / total discards by that player
        #   later in the discard sequence → higher value (more recent / reflects current intent)
        for p in range(4):
            ch = self.OFFSET_OBS['DISCARD_POS'] + p
            total = len(self.history[p])
            if total == 0:
                continue
            last_idx = {}
            for idx, tile in enumerate(self.history[p]):
                last_idx[tile] = idx  # keep only most-recent occurrence
            for tile, idx in last_idx.items():
                if tile in self.OFFSET_TILE:
                    self.obs[ch, self.OFFSET_TILE[tile]] = (idx + 1) / total

    def _get_pack_tile_counts(self, player):
        '''Return {tile: count} for all tiles locked in player's melds.'''
        d = defaultdict(int)
        for pack_type, tile, offer in self.packs[player]:
            if pack_type == 'CHI':
                color = tile[0]
                num = int(tile[1])
                for i in range(-1, 2):
                    d[color + str(num + i)] += 1
            elif pack_type == 'PENG':
                d[tile] += 3
            elif pack_type == 'GANG':
                if tile != 'CONCEALED':
                    d[tile] += 4
        return d

    def _count_embedding(self, offset, counts):
        for tile, count in counts.items():
            if tile in self.OFFSET_TILE:
                self.obs[offset : offset + min(4, count), self.OFFSET_TILE[tile]] = 1
    
    def _check_mahjong(self, winTile, isSelfDrawn = False, isAboutKong = False):
        try:
            fans = MahjongFanCalculator(
                pack = tuple(self.packs[0]),
                hand = tuple(self.hand),
                winTile = winTile,
                flowerCount = 0,
                isSelfDrawn = isSelfDrawn,
                is4thTile = (self.shownTiles[winTile] + isSelfDrawn) == 4,
                isAboutKong = isAboutKong,
                isWallLast = self.wallLast,
                seatWind = self.seatWind,
                prevalentWind = self.prevalentWind,
                verbose = True
            )
            fanCnt = 0
            for fanPoint, cnt, fanName, fanNameEn in fans:
                fanCnt += fanPoint * cnt
            if fanCnt < 8: raise Exception('Not Enough Fans')
        except:
            return False
        return True

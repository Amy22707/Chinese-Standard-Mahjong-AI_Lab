# Agent part
from feature import FeatureAgent

# Model part
from model import CNNModel

# Botzone interaction
import numpy as np
import torch
import os

USE_AUX_RANK = os.environ.get('USE_AUX_RANK', '0') == '1'
POSTPROCESS_MODE = os.environ.get('POSTPROCESS_MODE', 'light')


def _discard_danger(agent, tile):
    danger = 0.0
    for p in range(1, 4):
        danger = max(danger, agent._estimate_discard_danger(p, tile))
    return danger


def _attack_context(agent):
    ctx = {
        'shanten': 3,
        'chiitoi_shanten': 6,
        'best_flush_shanten': 13,
        'high_value': False,
        'pair_tiles': set(),
    }
    if not hasattr(agent, 'hand') or not hasattr(agent, 'packs'):
        return ctx
    try:
        from MahjongGB import MahjongShanten
        from collections import Counter

        shanten = MahjongShanten(hand = tuple(agent.hand), pack = tuple(agent.packs[0]))
        hand_cnt = Counter(agent.hand)
        pairs = sum(1 for c in hand_cnt.values() if c >= 2)
        chiitoi_shanten = max(0, 6 - pairs)

        best_flush_shanten = 13
        for suit in 'WTB':
            suit_tiles = [t for t in agent.hand if t[0] == suit]
            for pack_type, tile, _ in agent.packs[0]:
                if tile[0] == suit:
                    if pack_type == 'CHI':
                        num = int(tile[1])
                        suit_tiles.extend([suit + str(num - 1), tile, suit + str(num + 1)])
                    elif pack_type == 'PENG':
                        suit_tiles.extend([tile] * 3)
                    elif pack_type == 'GANG':
                        suit_tiles.extend([tile] * 4)
            best_flush_shanten = min(best_flush_shanten, max(0, 13 - len(suit_tiles)))

        ctx.update({
            'shanten': max(0, shanten),
            'chiitoi_shanten': chiitoi_shanten,
            'best_flush_shanten': best_flush_shanten,
            'high_value': chiitoi_shanten <= 2 or best_flush_shanten <= 4,
            'pair_tiles': {tile for tile, cnt in hand_cnt.items() if cnt >= 2},
        })
    except Exception:
        pass
    return ctx


def _threat_context(agent):
    '''Estimate how much the table asks us to fold.

    Uses only public information: wall progress, opponents' open melds, and discard count.
    '''
    try:
        total_discards = sum(len(h) for h in agent.history)
        my_wall = agent.tileWall[0] if hasattr(agent, 'tileWall') else 21
        late_hand = total_discards >= 42 or my_wall <= 8
        very_late = total_discards >= 56 or my_wall <= 4
        max_opp_melds = max((len(agent.packs[p]) for p in range(1, 4)), default = 0)
        many_open_melds = max_opp_melds >= 2
        threat = 0.0
        if late_hand:
            threat += 0.35
        if very_late:
            threat += 0.30
        if many_open_melds:
            threat += 0.25
        return {
            'late': late_hand,
            'very_late': very_late,
            'open_threat': many_open_melds,
            'level': min(1.0, threat),
        }
    except Exception:
        return {'late': False, 'very_late': False, 'open_threat': False, 'level': 0.0}


def _push_level(shanten, high_value, threat_level):
    if shanten <= 0:
        return 1.0
    if shanten == 1 and high_value:
        return 0.85
    if shanten == 1:
        return 0.65 - 0.25 * threat_level
    if high_value:
        return 0.55 - 0.20 * threat_level
    return 0.35 - 0.30 * threat_level


def _postprocess_action(agent, logits, mask, aux = None):
    legal_mask = mask.astype(bool)
    if not legal_mask.any():
        return FeatureAgent.OFFSET_ACT['Pass']

    # Winning is always preferred once it is legal; the 8-fan check is already in FeatureAgent.
    if legal_mask[FeatureAgent.OFFSET_ACT['Hu']]:
        return FeatureAgent.OFFSET_ACT['Hu']
    if POSTPROCESS_MODE == 'none':
        legal_logits = logits.copy()
        legal_logits[~legal_mask] = -np.inf
        return int(legal_logits.argmax())

    adjusted = logits.copy()
    raw = logits.copy()
    play_begin = FeatureAgent.OFFSET_ACT['Play']
    chi_begin = FeatureAgent.OFFSET_ACT['Chi']
    peng_begin = FeatureAgent.OFFSET_ACT['Peng']
    gang_begin = FeatureAgent.OFFSET_ACT['Gang']
    angang_begin = FeatureAgent.OFFSET_ACT['AnGang']
    bugang_begin = FeatureAgent.OFFSET_ACT['BuGang']

    ctx = _attack_context(agent)
    threat = _threat_context(agent)
    shanten = ctx['shanten']
    shanten_factor = min(1.0, max(0, shanten) / 3.0)
    push = _push_level(shanten, ctx['high_value'], threat['level'])
    risk_coeff = (1.35 if ctx['high_value'] else 1.60) * (1.0 + 0.65 * threat['level']) * (1.15 - 0.45 * push)
    legal_logits = raw.copy()
    legal_logits[~legal_mask] = -np.inf
    raw_best = int(legal_logits.argmax())

    # Balanced risk control: still push good hands, but avoid obvious far-hand deals.
    for tile, idx in FeatureAgent.OFFSET_TILE.items():
        a = play_begin + idx
        if legal_mask[a]:
            danger = _discard_danger(agent, tile)
            adjusted[a] -= risk_coeff * danger * shanten_factor
            if danger <= 0.05:
                adjusted[a] += (0.16 + 0.10 * threat['level']) * (1.0 + (1.0 - shanten_factor))
            if USE_AUX_RANK and aux is not None and 'discard_rank' in aux and idx < 34:
                adjusted[a] += 0.08 * (float(aux['discard_rank'][idx]) - 0.5)

    # Genbutsu bonus: tiles already discarded by any opponent are absolutely safe.
    for p in range(1, 4):
        for tile in set(agent.history[p]):
            if tile in FeatureAgent.OFFSET_TILE:
                a = play_begin + FeatureAgent.OFFSET_TILE[tile]
                if legal_mask[a]:
                    adjusted[a] += 0.3

    # Slightly more aggressive than the old bot, but still keeps some defensive flexibility.
    if push < 0.45:
        chi_penalty, peng_penalty, gang_penalty, bugang_penalty = 0.45, 0.35, 0.30, 0.25
    elif shanten >= 2 and not ctx['high_value']:
        chi_penalty, peng_penalty, gang_penalty, bugang_penalty = 0.38, 0.30, 0.26, 0.22
    elif shanten == 1 or ctx['high_value']:
        chi_penalty, peng_penalty, gang_penalty, bugang_penalty = 0.25, 0.20, 0.16, 0.14
    else:
        chi_penalty, peng_penalty, gang_penalty, bugang_penalty = 0.12, 0.10, 0.08, 0.08
    adjusted[chi_begin:peng_begin] -= chi_penalty
    adjusted[peng_begin:gang_begin] -= peng_penalty
    adjusted[gang_begin:angang_begin] -= gang_penalty
    adjusted[bugang_begin:] -= bugang_penalty
    if threat['level'] >= 0.6 and shanten >= 1 and not ctx['high_value']:
        adjusted[chi_begin:peng_begin] -= 0.15
        adjusted[peng_begin:gang_begin] -= 0.12
        adjusted[gang_begin:angang_begin] -= 0.10
        adjusted[bugang_begin:] -= 0.10

    # Bonus for high-value Peng/Gang: restore penalty if holding 3+ copies of the tile.
    if hasattr(agent, 'hand') and hasattr(agent, 'curTile') and agent.curTile is not None:
        tile = agent.curTile
        if agent.hand.count(tile) >= 2 and tile in FeatureAgent.OFFSET_TILE:
            idx = FeatureAgent.OFFSET_TILE[tile]
            peng_action = peng_begin + idx
            if legal_mask[peng_action]:
                adjusted[peng_action] += 0.15
            gang_action = gang_begin + idx
            if legal_mask[gang_action]:
                adjusted[gang_action] += 0.10

    # Tenpai still attacks, but no longer ignores deal-in danger completely.
    if shanten == 0:
        adjusted[FeatureAgent.OFFSET_ACT['Pass']] -= 1.0

    # Seven-pairs awareness: avoid breaking pairs when chiitoi is close.
    if ctx['chiitoi_shanten'] <= shanten and ctx['chiitoi_shanten'] <= 2:
        for tile in ctx['pair_tiles']:
            if tile in FeatureAgent.OFFSET_TILE:
                a = play_begin + FeatureAgent.OFFSET_TILE[tile]
                if legal_mask[a]:
                    adjusted[a] -= 0.5
    adjusted[~legal_mask] = -np.inf
    final_action = int(adjusted.argmax())

    # Danger gate: fold only far hands, or low-value one-shanten hands, when a safe close choice exists.
    should_gate = push < 0.75
    if should_gate and play_begin <= final_action < chi_begin:
        final_tile = FeatureAgent.TILE_LIST[final_action - play_begin]
        final_danger = _discard_danger(agent, final_tile)
        danger_threshold = 0.82 - 0.18 * threat['level']
        if final_danger >= danger_threshold:
            close_margin = 0.85 if shanten >= 2 else 0.50
            if threat['level'] >= 0.6:
                close_margin += 0.25
            best_safe = None
            best_safe_logit = -np.inf
            for tile, idx in FeatureAgent.OFFSET_TILE.items():
                a = play_begin + idx
                if not legal_mask[a]:
                    continue
                if raw[a] < raw[final_action] - close_margin:
                    continue
                if _discard_danger(agent, tile) <= 0.05 and raw[a] > best_safe_logit:
                    best_safe = a
                    best_safe_logit = raw[a]
            if best_safe is not None:
                return int(best_safe)

    # Do not let heuristics override a clearly preferred non-discard action.
    if raw_best < play_begin or raw_best >= chi_begin:
        if legal_logits[raw_best] >= adjusted[final_action] + 0.25:
            return raw_best
    return final_action


def _fallback_draw_response(obs):
    legal_mask = obs['action_mask'].astype(bool)
    play_begin = FeatureAgent.OFFSET_ACT['Play']
    chi_begin = FeatureAgent.OFFSET_ACT['Chi']
    legal_plays = np.flatnonzero(legal_mask[play_begin:chi_begin])
    if len(legal_plays):
        return 'PLAY %s' % FeatureAgent.TILE_LIST[int(legal_plays[0])]
    if legal_mask[FeatureAgent.OFFSET_ACT['Hu']]:
        return 'HU'
    for begin, text in (
        (FeatureAgent.OFFSET_ACT['AnGang'], 'GANG %s'),
        (FeatureAgent.OFFSET_ACT['BuGang'], 'BUGANG %s'),
    ):
        legal_tiles = np.flatnonzero(legal_mask[begin:begin + 34])
        if len(legal_tiles):
            return text % FeatureAgent.TILE_LIST[int(legal_tiles[0])]
    return 'PASS'

def obs2response(model, obs):
    with torch.no_grad():
        input_dict = {
            'observation': torch.from_numpy(np.expand_dims(obs['observation'], 0)),
            'action_mask': torch.from_numpy(np.expand_dims(obs['action_mask'], 0))
        }
        if USE_AUX_RANK:
            logits, aux = model(input_dict, return_aux = True)
            aux_np = {k: v.numpy().reshape(-1) for k, v in aux.items()}
        else:
            logits = model(input_dict)
            aux_np = None
    action = _postprocess_action(agent, logits.numpy().flatten(), obs['action_mask'], aux_np)
    response = agent.action2response(action)
    return response

import sys

if __name__ == '__main__':
    model = CNNModel()
    data_dir = os.environ.get('MODEL_PATH', '/data/best9.pkl')
    state = torch.load(data_dir, map_location = torch.device('cpu'))
    if isinstance(state, dict) and 'model' in state:
        state = state['model']
    current = model.state_dict()
    compatible = {k: v for k, v in state.items() if k in current and current[k].shape == v.shape}
    skipped = len(state) - len(compatible)
    current.update(compatible)
    model.load_state_dict(current)
    if skipped:
        print('INFO skipped %d incompatible checkpoint tensors' % skipped, file = sys.stderr)
    model.eval()
    zimo = False
    angang = None
    input() # 1
    while True:
        request = input()
        while not request.strip(): request = input()
        request = request.split()
        if request[0] == '0':
            seatWind = int(request[1])
            agent = FeatureAgent(seatWind)
            zimo = False
            angang = None
            agent.request2obs('Wind %s' % request[2])
            print('PASS')
        elif request[0] == '1':
            agent.request2obs(' '.join(['Deal', *request[5:]]))
            print('PASS')
        elif request[0] == '2':
            obs = agent.request2obs('Draw %s' % request[1])
            response = obs2response(model, obs)
            response = response.split()
            if response[0] == 'Hu':
                print('HU')
            elif response[0] == 'Play':
                print('PLAY %s' % response[1])
            elif response[0] == 'Gang':
                print('GANG %s' % response[1])
                angang = response[1]
            elif response[0] == 'BuGang':
                print('BUGANG %s' % response[1])
            else:
                print(_fallback_draw_response(obs))
        elif request[0] == '3':
            p = int(request[1])
            if request[2] == 'DRAW':
                agent.request2obs('Player %d Draw' % p)
                zimo = True
                print('PASS')
            elif request[2] == 'GANG':
                if p == seatWind and angang:
                    agent.request2obs('Player %d AnGang %s' % (p, angang))
                elif zimo:
                    agent.request2obs('Player %d AnGang' % p)
                else:
                    agent.request2obs('Player %d Gang' % p)
                print('PASS')
            elif request[2] == 'BUGANG':
                obs = agent.request2obs('Player %d BuGang %s' % (p, request[3]))
                if p == seatWind:
                    print('PASS')
                else:
                    response = obs2response(model, obs)
                    if response == 'Hu':
                        print('HU')
                    else:
                        print('PASS')
            else:
                zimo = False
                if request[2] == 'CHI':
                    agent.request2obs('Player %d Chi %s' % (p, request[3]))
                elif request[2] == 'PENG':
                    agent.request2obs('Player %d Peng' % p)
                obs = agent.request2obs('Player %d Play %s' % (p, request[-1]))
                if p == seatWind:
                    print('PASS')
                else:
                    response = obs2response(model, obs)
                    response = response.split()
                    if response[0] == 'Hu':
                        print('HU')
                    elif response[0] == 'Pass':
                        print('PASS')
                    elif response[0] == 'Gang':
                        print('GANG')
                        angang = None
                    elif response[0] in ('Peng', 'Chi'):
                        obs = agent.request2obs('Player %d '% seatWind + ' '.join(response))
                        response2 = obs2response(model, obs)
                        print(' '.join([response[0].upper(), *response[1:], response2.split()[-1]]))
                        agent.request2obs('Player %d Un' % seatWind + ' '.join(response))
        print('>>>BOTZONE_REQUEST_KEEP_RUNNING<<<')
        sys.stdout.flush()

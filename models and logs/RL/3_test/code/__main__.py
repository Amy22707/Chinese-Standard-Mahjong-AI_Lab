# Agent part
from feature import FeatureAgent

# Model part
from model import CNNModel

# Botzone interaction
import numpy as np
import torch
import os


def _discard_danger(agent, tile):
    danger = 0.0
    for p in range(1, 4):
        danger = max(danger, agent._estimate_discard_danger(p, tile))
    return danger


def _postprocess_action(agent, logits, mask):
    # Winning is always preferred once it is legal; the 8-fan check is already in FeatureAgent.
    if mask[FeatureAgent.OFFSET_ACT['Hu']]:
        return FeatureAgent.OFFSET_ACT['Hu']

    adjusted = logits.copy()
    play_begin   = FeatureAgent.OFFSET_ACT['Play']
    chi_begin    = FeatureAgent.OFFSET_ACT['Chi']
    peng_begin   = FeatureAgent.OFFSET_ACT['Peng']
    gang_begin   = FeatureAgent.OFFSET_ACT['Gang']
    angang_begin = FeatureAgent.OFFSET_ACT['AnGang']
    bugang_begin = FeatureAgent.OFFSET_ACT['BuGang']

    # Shanten-aware danger penalty: the closer to tenpai, the less we penalise risky discards.
    shanten_factor = 1.0
    if hasattr(agent, 'hand') and hasattr(agent, 'packs'):
        try:
            from MahjongGB import MahjongShanten
            shanten = MahjongShanten(hand=tuple(agent.hand), pack=tuple(agent.packs[0]))
            shanten_factor = min(1.0, max(0, shanten) / 3.0)
        except Exception:
            pass

    # Prefer safer discards when model confidence is close.
    for tile, idx in FeatureAgent.OFFSET_TILE.items():
        a = play_begin + idx
        if mask[a]:
            danger = _discard_danger(agent, tile)
            adjusted[a] -= 2.0 * danger * shanten_factor
            if danger <= 0.05:
                adjusted[a] += 0.25 * (1.0 + (1.0 - shanten_factor))

    # Genbutsu bonus: tiles already discarded by any opponent are absolutely safe.
    for p in range(1, 4):
        for tile in set(agent.history[p]):
            if tile in FeatureAgent.OFFSET_TILE:
                a = play_begin + FeatureAgent.OFFSET_TILE[tile]
                if mask[a]:
                    adjusted[a] += 0.3

    # Slightly conservative meld policy: modest penalty so model still melds when confident.
    adjusted[chi_begin:peng_begin]   -= 0.45
    adjusted[peng_begin:gang_begin]  -= 0.35
    adjusted[gang_begin:angang_begin] -= 0.30
    adjusted[bugang_begin:]          -= 0.25

    # Bonus for high-value Peng/Gang: restore penalty if holding 3+ copies of the tile.
    if hasattr(agent, 'hand') and hasattr(agent, 'curTile') and agent.curTile is not None:
        tile = agent.curTile
        if agent.hand.count(tile) >= 2 and tile in FeatureAgent.OFFSET_TILE:
            idx = FeatureAgent.OFFSET_TILE[tile]
            peng_action = peng_begin + idx
            if mask[peng_action]:
                adjusted[peng_action] += 0.15
            gang_action = gang_begin + idx
            if mask[gang_action]:
                adjusted[gang_action] += 0.10

    # Tenpai aggression: when already tenpai, reduce genbutsu bias and suppress Pass.
    if hasattr(agent, 'hand') and hasattr(agent, 'packs'):
        try:
            from MahjongGB import MahjongShanten
            from collections import Counter
            shanten = MahjongShanten(hand=tuple(agent.hand), pack=tuple(agent.packs[0]))
            if shanten == 0:
                # Already tenpai: halve the genbutsu bonus so we don't fold on a winning hand.
                for p in range(1, 4):
                    for t in set(agent.history[p]):
                        if t in FeatureAgent.OFFSET_TILE:
                            a = play_begin + FeatureAgent.OFFSET_TILE[t]
                            if mask[a]:
                                adjusted[a] -= 0.15
                # Suppress Pass so bot actively pursues the win.
                adjusted[FeatureAgent.OFFSET_ACT['Pass']] -= 1.0

            # Seven-pairs awareness: penalise discarding a pair tile when chiitoi is viable.
            hand_cnt = Counter(agent.hand)
            pairs = sum(1 for c in hand_cnt.values() if c >= 2)
            chiitoi_shanten = 6 - pairs
            if chiitoi_shanten <= shanten and chiitoi_shanten <= 2:
                for tile, cnt in hand_cnt.items():
                    if cnt >= 2 and tile in FeatureAgent.OFFSET_TILE:
                        a = play_begin + FeatureAgent.OFFSET_TILE[tile]
                        if mask[a]:
                            adjusted[a] -= 0.5
        except Exception:
            pass

    adjusted[~mask.astype(bool)] = -100.0
    return int(adjusted.argmax())


def obs2response(model, obs):
    with torch.no_grad():
        # 【修改点 1】改掉双输出解包，只用一个变量 logits 接收单输出结果
        logits = model({
            'observation': torch.from_numpy(np.expand_dims(obs['observation'], 0)),
            'action_mask': torch.from_numpy(np.expand_dims(obs['action_mask'], 0))
        })
    # 【修改点 2】配合 torch.no_grad()，直接用 .numpy() 获取数组即可
    action = _postprocess_action(agent, logits.numpy().flatten(), obs['action_mask'])
    response = agent.action2response(action)
    return response

import sys

if __name__ == '__main__':
    model = CNNModel()
    data_dir = os.environ.get('MODEL_PATH', '/data/model349.pt')
    ckpt = torch.load(data_dir, map_location=torch.device('cpu'))
    
    # 【修改点 3】由于你的网络是单输出，且可能包含名为 'model' 的子组件，
    # 我们去掉复杂的 if 键名误判逻辑，改用安全检查或直接加载：
    if isinstance(ckpt, dict) and 'model' in ckpt and isinstance(ckpt['model'], dict) and not any(k.startswith('model.') for k in ckpt.keys()):
        # 只有在确认为强化学习大包裹字典时才取 ['model']
        model.load_state_dict(ckpt['model'])
    else:
        # 普通的 state_dict 直接整包加载
        model.load_state_dict(ckpt)
        
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
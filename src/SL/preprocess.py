from feature import FeatureAgent
import numpy as np
import json
import multiprocessing
import os
import argparse

MAX_SEQ_LEN = 80


# Per-match worker logic. Must be top-level so multiprocessing can pickle it.

def _parse_winner(score_tokens):
    try:
        scores = [float(x) for x in score_tokens[1:5]]
    except (ValueError, TypeError):
        return None, None
    if len(scores) != 4:
        return None, None
    best = max(scores)
    if scores.count(best) != 1:
        return None, None
    return scores.index(best), scores


def process_match(args):
    '''Process one match block and write data/matchid.npz. Returns sample count.'''
    matchid, lines, winner_only = args
    obs     = [[] for _ in range(4)]
    actions = [[] for _ in range(4)]
    risks   = [[] for _ in range(4)]
    seq_tiles = [[] for _ in range(4)]
    seq_players = [[] for _ in range(4)]
    agents  = [FeatureAgent(i) for i in range(4)]
    curTile = None
    last_play_sample = None
    discard_events = []

    def seq_snapshot(player):
        events = []
        for order, abs_player, tile in discard_events:
            if tile in FeatureAgent.OFFSET_TILE:
                rel_player = (abs_player + 4 - player) % 4
                events.append((order, rel_player, FeatureAgent.OFFSET_TILE[tile]))
        events = events[-MAX_SEQ_LEN:]
        tile_arr = np.full(MAX_SEQ_LEN, 34, dtype = np.int64)
        player_arr = np.zeros(MAX_SEQ_LEN, dtype = np.int64)
        start = MAX_SEQ_LEN - len(events)
        for j, (_, rel_player, tile_id) in enumerate(events):
            tile_arr[start + j] = tile_id
            player_arr[start + j] = rel_player
        return tile_arr, player_arr

    def append_decision(player, obs_obj):
        obs[player].append(obs_obj)
        actions[player].append(0)
        risks[player].append(np.zeros(34, dtype = np.float32))
        tile_arr, player_arr = seq_snapshot(player)
        seq_tiles[player].append(tile_arr)
        seq_players[player].append(player_arr)

    def pop_replace_action(player, action):
        actions[player].pop()
        actions[player].append(action)

    for line in lines:
        t = line.split()
        if not t:
            continue
        if t[0] == 'Wind':
            for agent in agents:
                agent.request2obs(line)
        elif t[0] == 'Player':
            p = int(t[1])
            if t[2] == 'Deal':
                agents[p].request2obs(' '.join(t[2:]))
            elif t[2] == 'Draw':
                for i in range(4):
                    if i == p:
                        append_decision(p, agents[p].request2obs(' '.join(t[2:])))
                    else:
                        agents[i].request2obs(' '.join(t[:3]))
            elif t[2] == 'Play':
                pop_replace_action(p, agents[p].response2action(' '.join(t[2:])))
                last_play_sample = (p, len(actions[p]) - 1, FeatureAgent.OFFSET_TILE.get(t[3], -1))
                curTile = t[3]
                discard_events.append((len(discard_events), p, curTile))
                for i in range(4):
                    if i == p:
                        agents[p].request2obs(line)
                    else:
                        append_decision(i, agents[i].request2obs(line))
            elif t[2] == 'Chi':
                pop_replace_action(p, agents[p].response2action('Chi %s %s' % (curTile, t[3])))
                for i in range(4):
                    if i == p:
                        append_decision(p, agents[p].request2obs('Player %d Chi %s' % (p, t[3])))
                    else:
                        agents[i].request2obs('Player %d Chi %s' % (p, t[3]))
            elif t[2] == 'Peng':
                pop_replace_action(p, agents[p].response2action('Peng %s' % t[3]))
                for i in range(4):
                    if i == p:
                        append_decision(p, agents[p].request2obs('Player %d Peng %s' % (p, t[3])))
                    else:
                        agents[i].request2obs('Player %d Peng %s' % (p, t[3]))
            elif t[2] == 'Gang':
                pop_replace_action(p, agents[p].response2action('Gang %s' % t[3]))
                for i in range(4):
                    agents[i].request2obs('Player %d Gang %s' % (p, t[3]))
            elif t[2] == 'AnGang':
                pop_replace_action(p, agents[p].response2action('AnGang %s' % t[3]))
                for i in range(4):
                    if i == p:
                        agents[p].request2obs('Player %d AnGang %s' % (p, t[3]))
                    else:
                        agents[i].request2obs('Player %d AnGang' % p)
            elif t[2] == 'BuGang':
                pop_replace_action(p, agents[p].response2action('BuGang %s' % t[3]))
                for i in range(4):
                    if i == p:
                        agents[p].request2obs('Player %d BuGang %s' % (p, t[3]))
                    else:
                        append_decision(i, agents[i].request2obs('Player %d BuGang %s' % (p, t[3])))
            elif t[2] == 'Hu':
                pop_replace_action(p, agents[p].response2action('Hu'))
                if last_play_sample is not None and last_play_sample[0] != p:
                    dealer, sample_idx, tile_id = last_play_sample
                    if 0 <= tile_id < 34 and 0 <= sample_idx < len(risks[dealer]):
                        risks[dealer][sample_idx][tile_id] = 1.0
            # Deal with Ignore clause
            if t[2] in ['Peng', 'Gang', 'Hu']:
                for k in range(5, 15, 5):
                    if len(t) > k:
                        p2 = int(t[k + 1])
                        if t[k + 2] == 'Chi':
                            pop_replace_action(p2, agents[p2].response2action('Chi %s %s' % (curTile, t[k + 3])))
                        elif t[k + 2] == 'Peng':
                            pop_replace_action(p2, agents[p2].response2action('Peng %s' % t[k + 3]))
                        elif t[k + 2] == 'Gang':
                            pop_replace_action(p2, agents[p2].response2action('Gang %s' % t[k + 3]))
                        elif t[k + 2] == 'Hu':
                            pop_replace_action(p2, agents[p2].response2action('Hu'))
                    else:
                        break
        elif t[0] == 'Score':
            winner, scores = _parse_winner(t)
            players = [winner] if winner_only and winner is not None else range(4)
            # Filter states where only one valid action exists (Pass)
            for i in range(4):
                if i not in players:
                    obs[i] = []
                    actions[i] = []
                    risks[i] = []
                    seq_tiles[i] = []
                    seq_players[i] = []
                    continue
                pairs = [(o, a, r, st, sp) for o, a, r, st, sp in zip(obs[i], actions[i], risks[i], seq_tiles[i], seq_players[i])
                         if o['action_mask'].sum() > 1]
                obs[i]     = [p[0] for p in pairs]
                actions[i] = [p[1] for p in pairs]
                risks[i]   = [p[2] for p in pairs]
                seq_tiles[i] = [p[3] for p in pairs]
                seq_players[i] = [p[4] for p in pairs]
            # Build per-sample weight: proportional to winner's score (proxy for fan count).
            # Scores can be negative; shift so minimum is 1, then log-scale to avoid dominance.
            if scores is not None:
                winner_score = scores[winner] if winner is not None else 8.0
                # clamp: minimum fan equivalent ~8, maximum ~128
                fan_proxy = max(8.0, min(128.0, winner_score))
                sample_weight = float(np.log2(fan_proxy / 8.0) + 1.0)  # 1.0 for 8-fan, ~4.0 for 128-fan
                fan_target = float(np.log2(fan_proxy / 8.0) / 4.0)      # 0..1 for 8..128
            else:
                sample_weight = 1.0
                fan_target = 0.0
            # Save
            count = sum(len(x) for x in obs)
            if count > 0:
                player_ids = [i for i in range(4) for _ in obs[i]]
                obs_arr = np.stack([x['observation'] for i in range(4) for x in obs[i]])
                hand_arr = obs_arr[:, FeatureAgent.OFFSET_OBS['HAND'] : FeatureAgent.OFFSET_OBS['HAND'] + 4].sum(axis = 1)
                hand_flat = hand_arr.reshape(count, 36)
                honour_cnt = (hand_flat[:, 27:34] > 0).sum(axis = 1)
                chiitoi = obs_arr[:, FeatureAgent.OFFSET_OBS['CHIITOI'], 0, 0]
                flush = obs_arr[:, FeatureAgent.OFFSET_OBS['FLUSH'] : FeatureAgent.OFFSET_OBS['FLUSH'] + 3, 0, 0]
                fan_route = np.zeros(count, dtype = np.int64)
                fan_route[chiitoi <= 2.0 / 6.0] = 1
                fan_route[np.min(flush, axis = 1) <= 4.0 / 13.0] = 2
                fan_route[(fan_route == 0) & (honour_cnt >= 4)] = 4
                np.savez(
                    'data/%d.npz' % matchid,
                    obs  = obs_arr.astype(np.float16),
                    mask = np.stack([x['action_mask']  for i in range(4) for x in obs[i]]).astype(np.int8),
                    act  = np.array([x                 for i in range(4) for x in actions[i]]),
                    wt   = np.full(count, sample_weight, dtype = np.float32),
                    win  = np.array([1.0 if i == winner else 0.0 for i in player_ids], dtype = np.float32),
                    fan  = np.full(count, fan_target, dtype = np.float32),
                    shanten = obs_arr[:, FeatureAgent.OFFSET_OBS['SHANTEN'], 0, 0].astype(np.float32),
                    risk = np.stack([x for i in range(4) for x in risks[i]]).astype(np.float32),
                    seq_tile = np.stack([x for i in range(4) for x in seq_tiles[i]]).astype(np.int64),
                    seq_player = np.stack([x for i in range(4) for x in seq_players[i]]).astype(np.int64),
                    fan_route = fan_route,
                )
            return count

    return 0   # malformed match (no Score line)


# Main entry

def split_matches(path):
    '''Read data.txt and return list of (matchid, lines) without Match/Score lines.'''
    blocks = []
    current = []
    with open(path, encoding='UTF-8') as f:
        for line in f:
            t = line.split()
            if not t:
                continue
            if t[0] == 'Match':
                current = []          # start new block (don't include the Match line itself)
            elif t[0] == 'Score':
                current.append(line)  # Score triggers save inside worker
                blocks.append(current)
                current = []
            else:
                current.append(line)
    return blocks


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--workers', type=int, default=max(1, os.cpu_count() - 1),
                        help='number of parallel worker processes (default: cpu_count-1)')
    parser.add_argument('--all-players', action='store_true',
                        help='keep all players instead of only the highest-score winner')
    args = parser.parse_args()

    os.makedirs('data', exist_ok=True)

    print('Splitting data.txt into match blocks...')
    blocks = split_matches('data/data.txt')
    total  = len(blocks)
    print('Total matches:', total)

    tasks = [(i, block, not args.all_players) for i, block in enumerate(blocks)]

    print('Processing with %d workers...' % args.workers)
    with multiprocessing.Pool(processes=args.workers) as pool:
        counts = []
        for i, cnt in enumerate(pool.imap(process_match, tasks, chunksize=16)):
            counts.append(cnt)
            if (i + 1) % 512 == 0:
                print('  processed %d / %d matches' % (i + 1, total))
    print('Done. Writing count.json...')
    with open('data/count.json', 'w') as f:
        json.dump(counts, f)

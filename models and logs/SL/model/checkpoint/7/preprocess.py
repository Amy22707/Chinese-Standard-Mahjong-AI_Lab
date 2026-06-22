from feature import FeatureAgent
import numpy as np
import json
import multiprocessing
import os
import argparse


# Per-match worker logic. Must be top-level so multiprocessing can pickle it.

def _parse_winner(score_tokens):
    try:
        scores = [float(x) for x in score_tokens[1:5]]
    except (ValueError, TypeError):
        return None
    if len(scores) != 4:
        return None
    best = max(scores)
    if scores.count(best) != 1:
        return None
    return scores.index(best)


def process_match(args):
    '''Process one match block and write data/matchid.npz. Returns sample count.'''
    matchid, lines, winner_only = args
    obs     = [[] for _ in range(4)]
    actions = [[] for _ in range(4)]
    agents  = [FeatureAgent(i) for i in range(4)]
    curTile = None

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
                        obs[p].append(agents[p].request2obs(' '.join(t[2:])))
                        actions[p].append(0)
                    else:
                        agents[i].request2obs(' '.join(t[:3]))
            elif t[2] == 'Play':
                actions[p].pop()
                actions[p].append(agents[p].response2action(' '.join(t[2:])))
                for i in range(4):
                    if i == p:
                        agents[p].request2obs(line)
                    else:
                        obs[i].append(agents[i].request2obs(line))
                        actions[i].append(0)
                curTile = t[3]
            elif t[2] == 'Chi':
                actions[p].pop()
                actions[p].append(agents[p].response2action('Chi %s %s' % (curTile, t[3])))
                for i in range(4):
                    if i == p:
                        obs[p].append(agents[p].request2obs('Player %d Chi %s' % (p, t[3])))
                        actions[p].append(0)
                    else:
                        agents[i].request2obs('Player %d Chi %s' % (p, t[3]))
            elif t[2] == 'Peng':
                actions[p].pop()
                actions[p].append(agents[p].response2action('Peng %s' % t[3]))
                for i in range(4):
                    if i == p:
                        obs[p].append(agents[p].request2obs('Player %d Peng %s' % (p, t[3])))
                        actions[p].append(0)
                    else:
                        agents[i].request2obs('Player %d Peng %s' % (p, t[3]))
            elif t[2] == 'Gang':
                actions[p].pop()
                actions[p].append(agents[p].response2action('Gang %s' % t[3]))
                for i in range(4):
                    agents[i].request2obs('Player %d Gang %s' % (p, t[3]))
            elif t[2] == 'AnGang':
                actions[p].pop()
                actions[p].append(agents[p].response2action('AnGang %s' % t[3]))
                for i in range(4):
                    if i == p:
                        agents[p].request2obs('Player %d AnGang %s' % (p, t[3]))
                    else:
                        agents[i].request2obs('Player %d AnGang' % p)
            elif t[2] == 'BuGang':
                actions[p].pop()
                actions[p].append(agents[p].response2action('BuGang %s' % t[3]))
                for i in range(4):
                    if i == p:
                        agents[p].request2obs('Player %d BuGang %s' % (p, t[3]))
                    else:
                        obs[i].append(agents[i].request2obs('Player %d BuGang %s' % (p, t[3])))
                        actions[i].append(0)
            elif t[2] == 'Hu':
                actions[p].pop()
                actions[p].append(agents[p].response2action('Hu'))
            # Deal with Ignore clause
            if t[2] in ['Peng', 'Gang', 'Hu']:
                for k in range(5, 15, 5):
                    if len(t) > k:
                        p2 = int(t[k + 1])
                        if t[k + 2] == 'Chi':
                            actions[p2].pop()
                            actions[p2].append(agents[p2].response2action('Chi %s %s' % (curTile, t[k + 3])))
                        elif t[k + 2] == 'Peng':
                            actions[p2].pop()
                            actions[p2].append(agents[p2].response2action('Peng %s' % t[k + 3]))
                        elif t[k + 2] == 'Gang':
                            actions[p2].pop()
                            actions[p2].append(agents[p2].response2action('Gang %s' % t[k + 3]))
                        elif t[k + 2] == 'Hu':
                            actions[p2].pop()
                            actions[p2].append(agents[p2].response2action('Hu'))
                    else:
                        break
        elif t[0] == 'Score':
            winner = _parse_winner(t)
            players = [winner] if winner_only and winner is not None else range(4)
            # Filter states where only one valid action exists (Pass)
            for i in range(4):
                if i not in players:
                    obs[i] = []
                    actions[i] = []
                    continue
                pairs = [(o, a) for o, a in zip(obs[i], actions[i]) if o['action_mask'].sum() > 1]
                obs[i]     = [p[0] for p in pairs]
                actions[i] = [p[1] for p in pairs]
            # Save
            count = sum(len(x) for x in obs)
            if count > 0:
                obs_arr = np.stack([x['observation'] for i in range(4) for x in obs[i]])
                np.savez(
                    'data/%d.npz' % matchid,
                    obs  = obs_arr.astype(np.float16),  # float16: preserves continuous features (WALL, DISCARD_POS, etc.)
                    mask = np.stack([x['action_mask']  for i in range(4) for x in obs[i]]).astype(np.int8),
                    act  = np.array([x                 for i in range(4) for x in actions[i]])
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

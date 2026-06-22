from multiprocessing import Process, Event
import numpy as np
import torch
import signal
import random

from replay_buffer import ReplayBuffer
from model_pool import ModelPoolClient
from env import MahjongGBEnv
from feature import FeatureAgent
from model import CNNModel

class Actor(Process):
    
    def __init__(self, config, replay_buffer):
        super(Actor, self).__init__()
        self.replay_buffer = replay_buffer
        self.config = config
        self.name = config.get('name', 'Actor-?')
        self.stop_event = Event()

    def stop(self):
        self.stop_event.set()
        
    def run(self):
        # keep signal handling in main process only
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        signal.signal(signal.SIGTERM, lambda signum, frame: self.stop_event.set())
        torch.set_num_threads(1)
    
        # connect to model pool
        model_pool = ModelPoolClient(self.config['model_pool_name'])
        try:
            # Hero model: the model being trained (always latest)
            hero_model = CNNModel()
            # Opponent model: uses random historical models for diversity
            opp_model = CNNModel()
            
            # load initial model
            version = model_pool.get_latest_model()
            state_dict = model_pool.load_model(version)
            hero_model.load_state_dict(state_dict)
            opp_model.load_state_dict(state_dict)
            
            # collect data
            env = MahjongGBEnv(config = self.config.get('env_config', {'agent_clz': FeatureAgent}))
            # Probability of using a random historical model for opponents (0=always latest, 1=always historical)
            opp_historical_prob = self.config.get('opp_historical_prob', 0.8)
            
            for episode in range(self.config['episodes_per_actor']):
                if self.stop_event.is_set():
                    break
                # Update hero model to the latest version
                latest = model_pool.get_latest_model()
                if latest['id'] > version['id']:
                    state_dict = model_pool.load_model(latest)
                    hero_model.load_state_dict(state_dict)
                    version = latest
                
                # Opponent model: randomly pick a historical version for diversity.
                # This prevents self-play policy collapse by exposing hero to diverse opponents.
                model_list = model_pool.get_model_list()
                if len(model_list) > 1 and random.random() < opp_historical_prob:
                    opp_version = random.choice(model_list[:-1])  # exclude latest
                    opp_state = model_pool.load_model(opp_version)
                    if opp_state is not None:
                        opp_model.load_state_dict(opp_state)
                    else:
                        opp_model.load_state_dict(state_dict)
                else:
                    opp_model.load_state_dict(state_dict)

                # Hero seat rotates across episodes so all positions get trained equally
                hero_seat = episode % 4
                hero_name = env.agent_names[hero_seat]

                # run one episode and collect trajectory data for hero seat only
                obs = env.reset()
                hero_data = {
                    'state' : {'observation': [], 'action_mask': []},
                    'action'   : [],
                    'log_prob' : [],
                    'reward'   : [],
                    'value'    : []
                }
                done = False
                hero_model.train(False)
                opp_model.train(False)
                while not done and not self.stop_event.is_set():
                    actions = {}
                    for agent_name in obs:
                        state = obs[agent_name]
                        seat = env.agent_names.index(agent_name)
                        active_model = hero_model if agent_name == hero_name else opp_model
                        obs_t  = torch.tensor(state['observation'], dtype=torch.float).unsqueeze(0)
                        mask_t = torch.tensor(state['action_mask'],  dtype=torch.float).unsqueeze(0)
                        with torch.no_grad():
                            logits, value = active_model({'observation': obs_t, 'action_mask': mask_t})
                            action_dist = torch.distributions.Categorical(logits=logits)
                            action = action_dist.sample()
                            log_prob = action_dist.log_prob(action).item()
                            action = action.item()
                            value = value.item()
                        actions[agent_name] = action
                        # Only store trajectory data for hero seat
                        if agent_name == hero_name:
                            hero_data['state']['observation'].append(state['observation'])
                            hero_data['state']['action_mask'].append(state['action_mask'])
                            hero_data['action'].append(action)
                            hero_data['log_prob'].append(log_prob)
                            hero_data['value'].append(value)
                    # interact with env
                    next_obs, rewards, done = env.step(actions)
                    if hero_name in rewards:
                        hero_data['reward'].append(rewards[hero_name])
                    obs = next_obs
                if self.stop_event.is_set():
                    break
                print(self.name, 'Episode', episode,
                      'HeroSeat', hero_seat,
                      'OppModel', opp_version['id'] if len(model_list) > 1 else version['id'],
                      'Model', latest['id'],
                      'Reward', rewards, flush=True)
                
                # Post-process hero trajectory: compute GAE advantages
                agent_data = hero_data
                if len(agent_data['action']) < len(agent_data['reward']):
                    agent_data['reward'].pop(0)
                if len(agent_data['action']) == 0:
                    continue
                obs_arr   = np.stack(agent_data['state']['observation'])
                mask_arr  = np.stack(agent_data['state']['action_mask'])
                actions_arr = np.array(agent_data['action'],   dtype=np.int64)
                log_probs   = np.array(agent_data['log_prob'], dtype=np.float32)
                rewards_arr = np.array(agent_data['reward'],   dtype=np.float32)
                values_arr  = np.array(agent_data['value'],    dtype=np.float32)
                next_values = np.array(agent_data['value'][1:] + [0], dtype=np.float32)
                
                # GAE (Generalised Advantage Estimation)
                gamma  = self.config['gamma']
                lam    = self.config['lambda']
                td_target = rewards_arr + gamma * next_values
                td_delta  = td_target - values_arr
                advs, adv = [], 0.0
                for delta in td_delta[::-1]:
                    adv = gamma * lam * adv + delta
                    advs.append(adv)
                advs.reverse()
                advantages = np.array(advs, dtype=np.float32)
                
                payload = {
                    'state': {'observation': obs_arr, 'action_mask': mask_arr},
                    'action'   : actions_arr,
                    'log_prob' : log_probs,
                    'value'    : values_arr,
                    'adv'      : advantages,
                    'target'   : td_target
                }
                while not self.stop_event.is_set():
                    if self.replay_buffer.push(payload, timeout=0.5):
                        break
        finally:
            model_pool.close()

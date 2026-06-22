from multiprocessing import Process, Event
import numpy as np
import torch
import signal

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
            # create network model
            model = CNNModel()
            
            # load initial model
            version = model_pool.get_latest_model()
            state_dict = model_pool.load_model(version)
            model.load_state_dict(state_dict)
            
            # collect data
            env = MahjongGBEnv(config = self.config.get('env_config', {'agent_clz': FeatureAgent}))
            
            for episode in range(self.config['episodes_per_actor']):
                if self.stop_event.is_set():
                    break
                # update model to the latest version
                latest = model_pool.get_latest_model()
                if latest['id'] > version['id']:
                    state_dict = model_pool.load_model(latest)
                    model.load_state_dict(state_dict)
                    version = latest
                
                # run one episode and collect trajectory data
                obs = env.reset()
                episode_data = {agent_name: {
                    'state' : {
                        'observation': [],
                        'action_mask': []
                    },
                    'action'   : [],
                    'log_prob' : [],   # log π(a|s) at collection time for PPO importance ratio
                    'reward'   : [],
                    'value'    : []
                } for agent_name in env.agent_names}
                done = False
                model.train(False)  # BatchNorm inference mode – set once per episode
                while not done and not self.stop_event.is_set():
                    actions = {}
                    for agent_name in obs:
                        agent_data = episode_data[agent_name]
                        state = obs[agent_name]
                        agent_data['state']['observation'].append(state['observation'])
                        agent_data['state']['action_mask'].append(state['action_mask'])
                        obs_t = torch.tensor(state['observation'], dtype=torch.float).unsqueeze(0)
                        mask_t = torch.tensor(state['action_mask'], dtype=torch.float).unsqueeze(0)
                        with torch.no_grad():
                            logits, value = model({'observation': obs_t, 'action_mask': mask_t})
                            action_dist = torch.distributions.Categorical(logits=logits)
                            action = action_dist.sample()
                            log_prob = action_dist.log_prob(action).item()
                            action = action.item()
                            value = value.item()
                        actions[agent_name] = action
                        agent_data['action'].append(action)
                        agent_data['log_prob'].append(log_prob)
                        agent_data['value'].append(value)
                    # interact with env
                    next_obs, rewards, done = env.step(actions)
                    for agent_name in rewards:
                        episode_data[agent_name]['reward'].append(rewards[agent_name])
                    obs = next_obs
                if self.stop_event.is_set():
                    break
                print(self.name, 'Episode', episode, 'Model', latest['id'], 'Reward', rewards, flush=True)
                
                # Post-process episode data: compute GAE advantages per agent
                for agent_name, agent_data in episode_data.items():
                    if len(agent_data['action']) < len(agent_data['reward']):
                        agent_data['reward'].pop(0)
                    obs_arr   = np.stack(agent_data['state']['observation'])
                    mask_arr  = np.stack(agent_data['state']['action_mask'])
                    actions   = np.array(agent_data['action'],   dtype=np.int64)
                    log_probs = np.array(agent_data['log_prob'], dtype=np.float32)
                    rewards   = np.array(agent_data['reward'],   dtype=np.float32)
                    values    = np.array(agent_data['value'],    dtype=np.float32)
                    next_values = np.array(agent_data['value'][1:] + [0], dtype=np.float32)
                    
                    # GAE (Generalised Advantage Estimation)
                    gamma  = self.config['gamma']
                    lam    = self.config['lambda']
                    td_target = rewards + gamma * next_values
                    td_delta  = td_target - values
                    advs, adv = [], 0.0
                    for delta in td_delta[::-1]:
                        adv = gamma * lam * adv + delta
                        advs.append(adv)
                    advs.reverse()
                    advantages = np.array(advs, dtype=np.float32)
                    
                    payload = {
                        'state': {
                            'observation': obs_arr,
                            'action_mask': mask_arr
                        },
                        'action'   : actions,
                        'log_prob' : log_probs,
                        'value'    : values,     # stored critic estimates for PPO value clipping
                        'adv'      : advantages,
                        'target'   : td_target
                    }
                    while not self.stop_event.is_set():
                        if self.replay_buffer.push(payload, timeout=0.5):
                            break
        finally:
            model_pool.close()

import os
# 设置OMP_WAIT_POLICY为PASSIVE，让等待的线程不消耗CPU资源 #确保在pytorch前设置
os.environ['OMP_WAIT_POLICY'] = 'PASSIVE' #确保在pytorch前设置

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

import numpy as np
from Buffer import Buffer # 与DQN.py中的Buffer一样

from copy import deepcopy
import pettingzoo #动态导入
# 自定义环境
from env import ToyEnv
import math

import gymnasium as gym
import importlib
import argparse
from torch.utils.tensorboard import SummaryWriter
import time

## 第一部分：定义Agent类
class Actor(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_1=128, hidden_2=128):
        super(Actor, self).__init__()
        self.l1 = nn.Linear(obs_dim, hidden_1)
        self.l2 = nn.Linear(hidden_1, hidden_2)
        self.l3 = nn.Linear(hidden_2, action_dim)
    
    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))

        return F.tanh(self.l3(x))
    
class Critic(nn.Module):
    def __init__(self, dim_info:dict, hidden_1=128 , hidden_2=128):
        super(Critic, self).__init__()
        global_obs_act_dim = sum(sum(val) for val in dim_info.values())  
        
        self.l1 = nn.Linear(global_obs_act_dim, hidden_1)
        self.l2 = nn.Linear(hidden_1, hidden_2)
        self.l3 = nn.Linear(hidden_2, 1)

    def forward(self, s, a): # 传入全局观测和动作
        sa = torch.cat(list(s)+list(a), dim = 1)
        #sa = torch.cat([s,a], dim = 1)
        
        q = F.relu(self.l1(sa))
        q = F.relu(self.l2(q))
        q = self.l3(q)

        return q

class Agent:
    def __init__(self, obs_dim, action_dim, dim_info,actor_lr, critic_lr, device):
        
        self.actor = Actor(obs_dim, action_dim, )
        self.critic = Critic( dim_info )

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)

        self.actor_target = deepcopy(self.actor)
        self.critic_target = deepcopy(self.critic)

    def update_actor(self, loss):
        self.actor_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
        self.actor_optimizer.step()

    def update_critic(self, loss):
        self.critic_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
        self.critic_optimizer.step()


## 第二部分：定义DQN算法类
class MADDPG: 
    def __init__(self, dim_info, is_continue, actor_lr, critic_lr, buffer_size, device, trick = None):

        self.agents  = {}
        self.buffers = {}
        for agent_id, (obs_dim, action_dim) in dim_info.items():
            self.agents[agent_id] = Agent(obs_dim, action_dim, dim_info, actor_lr, critic_lr, device=device)
            self.buffers[agent_id] = Buffer(buffer_size, obs_dim, act_dim = action_dim if is_continue else 1, device = 'cpu')

        self.device = device
        self.is_continue = is_continue
        self.agent_x = list(self.agents.keys())[0] #sample 用

    def select_action(self, obs):
        actions = {}
        for agent_id, obs in obs.items():
            obs = torch.as_tensor(obs,dtype=torch.float32).reshape(1, -1).to(self.device)
            ''' 区别2 '''
            if self.is_continue: # dqn 无此项
                action = self.agents[agent_id].actor(obs)
                actions[agent_id] = action.detach().cpu().numpy().squeeze(0) # 1xaction_dim -> action_dim
            else:
                action = self.agents[agent_id].argmax(dim = 1).detach().cpu().numpy()[0] # []标量
                actions[agent_id] = action
        return actions
    
    def add(self, obs, action, reward, next_obs, done):
        for agent_id, buffer in self.buffers.items():
            buffer.add(obs[agent_id], action[agent_id], reward[agent_id], next_obs[agent_id], done[agent_id])

    def sample(self, batch_size):
        total_size = len(self.buffers[self.agent_x])
        indices = np.random.choice(total_size, batch_size, replace=False)

        obs, action, reward, next_obs, done = {}, {}, {}, {}, {}
        for agent_id, buffer in self.buffers.items():
            obs[agent_id], action[agent_id], reward[agent_id], next_obs[agent_id], done[agent_id] = buffer.sample(indices)

        return obs, action, reward, next_obs, done #包含所有智能体的数据
    
    ## DDPG算法相关
    def learn(self, batch_size ,gamma , tau):
        # 多智能体特有-- 集中式训练critic:计算next_q值时,要用到所有智能体next状态和动作
        for agent_id, agent in self.agents.items():
            ## 更新前准备
            obs, action, reward, next_obs, done = self.sample(batch_size) # 必须放for里，否则报二次传播错，原因是原来的数据在计算图中已经被释放了
            ''' 区别1 '''
            next_action = {}
            for agent_id_, agent_ in self.agents.items():
                next_action_i = agent_.actor_target(next_obs[agent_id_]) 
                next_action[agent_id_] = next_action_i
            next_target_Q = agent.critic_target(next_obs.values(), next_action.values())
            
            # 先更新critic
            target_Q = reward[agent_id] + gamma * next_target_Q * (1 - done[agent_id])
            current_Q = agent.critic(obs.values(), action.values())
            critic_loss = F.mse_loss(current_Q, target_Q.detach())
            agent.update_critic(critic_loss)

            # 再更新actor
            new_action = agent.actor(obs[agent_id])
            action[agent_id] = new_action
            actor_loss = -agent.critic(obs.values(), action.values()).mean()
            agent.update_actor(actor_loss)
        
        self.update_target(tau)

    def update_target(self, tau):
        def soft_update(target, source, tau):
            for target_param, param in zip(target.parameters(), source.parameters()):
                target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)
            
        for agent in self.agents.values():
            soft_update(agent.actor_target, agent.actor, tau)
            soft_update(agent.critic_target, agent.critic, tau)


    def save(self, model_path):
        torch.save(
            {name: agent.actor.state_dict() for name, agent in self.agents.items()},
            os.path.join(model_path, 'MADDPG.pth')
        )
        
    ## 加载模型
    @staticmethod 
    def load(dim_info, is_continue, model_dir):
        policy = MADDPG(dim_info, is_continue = is_continue, actor_lr = 0, critic_lr = 0, buffer_size = 0, device = 'cpu')
        torch.load(
            os.path.join(model_dir, 'MADDPG.pth')
        )
        return policy
    
## 第三部分 mian 函数
## 环境配置
def get_env(env_name,render_mode=False):
    if env_name == 'toy_env':
        env = ToyEnv(render_mode=render_mode)
    env.reset()
    dim_info = {} # dict{agent_id:[obs_dim,action_dim]}
    for agent_id in env.agents:
        dim_info[agent_id] = []
        if isinstance(env.observation_space(agent_id), gym.spaces.Box):
            dim_info[agent_id].append(env.observation_space(agent_id).shape[0])
        else:
            dim_info[agent_id].append(1)
        if isinstance(env.action_space(agent_id), gym.spaces.Box):
            dim_info[agent_id].append(env.action_space(agent_id).shape[0])
        else:
            dim_info[agent_id].append(env.action_space(agent_id).n)

    return env,dim_info, 2*math.pi, True # pettingzoo.mpe 环境中，max_action均为1 , 选取连续环境is_continue = True

## make_dir 与DQN.py 里一样
def make_dir(env_name,policy_name = 'DQN',trick = None):
    script_dir = os.path.dirname(os.path.abspath(__file__)) # 当前脚本文件夹
    env_dir = os.path.join(script_dir,'./results', env_name)
    os.makedirs(env_dir) if not os.path.exists(env_dir) else None
    print('trick:',trick)
    # 确定前缀
    if trick is None or not any(trick.values()):
        prefix = policy_name + '_'
    else:
        prefix = policy_name + '_'
        for key in trick.keys():
            if trick[key]:
                prefix += key + '_'
    # 查找现有的文件夹并确定下一个编号
    existing_dirs = [d for d in os.listdir(env_dir) if d.startswith(prefix) and d[len(prefix):].isdigit()]
    max_number = 0 if not existing_dirs else max([int(d.split('_')[-1]) for d in existing_dirs if d.split('_')[-1].isdigit()])
    model_dir = os.path.join(env_dir, prefix + str(max_number + 1))
    os.makedirs(model_dir)
    return model_dir

'''
1: win_rate_adjust 0.03
2: win_rate_adjust 0.05
'''
''' 环境见
simple_adversary_v3,simple_crypto_v3,simple_push_v3,simple_reference_v3,simple_speaker_listener_v3,simple_spread_v3,simple_tag_v3
https://pettingzoo.farama.org/environments/mpe
'''
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # 环境参数
    parser.add_argument("--env_name", type = str,default="toy_env") 
    parser.add_argument("--N", type=int, default=5) # 环境中智能体数量 默认None 这里用来对比设置 暂无此参数
    parser.add_argument("--max_action", type=float, default=2*math.pi)
    # 共有参数
    parser.add_argument("--seed", type=int, default=0) # 0 10 100
    parser.add_argument("--max_episodes", type=int, default=int(200000))  # 得改成 200000 
    parser.add_argument("--start_steps", type=int, default=50000) # 满足此开始更新
    parser.add_argument("--random_steps", type=int, default=50000)  #dqn 无此参数 满足此开始自己探索
    parser.add_argument("--learn_steps_interval", type=int, default=1)
    parser.add_argument("--learn_interval", type=int, default=3) #episode
    # 训练参数
    parser.add_argument("--gamma", type=float, default=0.95)
    parser.add_argument("--tau", type=float, default=0.01)
    ## AC参数
    parser.add_argument("--actor_lr", type=float, default=1e-3)
    parser.add_argument("--critic_lr", type=float, default=1e-3)
    ## buffer参数   
    parser.add_argument("--buffer_size", type=int, default=1e6) #1e6默认是float,在bufffer中有int强制转换
    parser.add_argument("--batch_size", type=int, default=1024)  #保证比start_steps小
    # DDPG 独有参数 noise
    parser.add_argument("--gauss_sigma", type=float, default=0.1)
    # trick参数
    parser.add_argument("--policy_name", type=str, default='MADDPG')
    parser.add_argument("--trick", type=dict, default=None)  
    # 环境参数
    parser.add_argument('--policy_number', type=int, default=0, help='number of policy')

    parser.add_argument('--policy_noise_a', type=float, default=0.05*2*math.pi, help='policy noise 1') # 此时没用到
    parser.add_argument('--policy_noise_b', type=float, default=0.12*2*math.pi, help='policy noise 1') # 此时没用到

    parser.add_argument('--win_rate_adjust', type=float, default=0.03, help='win rate adjust')  # 0.05 3v3 0

    parser.add_argument('--noise_std', type=float, default=0.1*2*math.pi, help='std of noise')

    args = parser.parse_args()

    ## 环境配置
    env,dim_info,max_action,is_continue = get_env(args.env_name)
    max_action = max_action if max_action is not None else args.max_action

    ## 随机数种子
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    ## 保存model文件夹
    model_dir = make_dir(args.env_name,policy_name = args.policy_name,trick=args.trick)
    writer = SummaryWriter(model_dir)

    ##
    device = torch.device('cpu')

    ## 算法配置
    policy = MADDPG(dim_info, is_continue, args.actor_lr, args.critic_lr, args.buffer_size, device, args.trick)

    time_ = time.time()
    # 环境相关 
    win_list = [] #一般胜
    win1_list = [] #大奖励

    red_tank_0_hp_l = []
    red_tank_1_hp_l = []
    red_tank_2_hp_l = []
    policy_up = 0
    ## 训练
    episode_num = 0
    step = 0
    env_agents = [agent_id for agent_id in env.agents]
    episode_reward =  {agent_id: np.zeros(args.max_episodes) for agent_id in env.agents} #{agent_id: 0 for agent_id in env_agents}
    train_return = {agent_id: [] for agent_id in env_agents}
    obs,info = env.reset(args.policy_number)
    obs = env.Normalization(obs) #状态归一


    for episode in range(args.max_episodes):
        obs, infos = env.reset(args.policy_number,args.policy_noise_a,args.policy_noise_b) #改1
        obs = env.Normalization(obs) #状态归一
        agent_reward = {agent_id: 0 for agent_id in env.agents}  # agent reward of the current episode

        done = {agent_id: False for agent_id in env.agents}
        while not any(done.values()):  # interact with the env for an episode
            step += 1
            if step < args.random_steps: #此时action为(-1,1)
                action_nor = env.sample()# 也可以{agent_id: env.action_space(agent_id).sample() for agent_id in env.agents}
            else:
                action_nor = policy.select_action(obs)
            
            if args.policy_name == 'MATD3' or 'MADDPG':
                # 加噪音
                action = {agent_id: [np.clip(a[0]*args.max_action + np.random.normal(0, args.noise_std,), 
                                -args.max_action, args.max_action)]
                                for agent_id, a in action_nor.items()}
            elif args.policy_name == 'MAAC':
                action = {agent_id: [np.clip(a[0]*args.max_action, -args.max_action, args.max_action)]
                                for agent_id, a in action_nor.items()}
            
            ## 单个值就行？ 还是得列表？ 都是列表的形式
            #print('action:',action_nor)
            #print('action:',action)
            next_obs, reward, terminations, truncations, info = env.step(action)
            next_obs = env.Normalization(next_obs) #状态归一
            
            done ={agent_id: terminations[agent_id] or truncations[agent_id] for agent_id in env.agents}

            if any(done.values()):
                #每个时间步惩罚和结算奖励  
                for agent_id, r in reward.items():
                    #reward[agent_id] -= 0.01 # 暂时取消

                    if info['win1'] == True: #只有结算时有 大win
                        reward[agent_id] +=  10  #3*r  #100
                        if info[agent_id] > 1e-3: 
                            reward[agent_id] +=  3  #存活奖励
                            reward[agent_id] +=  info[agent_id]*3 #生命值奖励

            policy.add(obs, action_nor, reward, next_obs, done) # 增加一步的经验

            for agent_id, r in reward.items():  # update reward
                agent_reward[agent_id] += r
                
            obs = next_obs

        # episode finishes
        if step >= args.random_steps and episode % args.learn_interval == 0:  # learn every few steps
            if args.policy_name == 'MADDPG':
                policy.learn(args.batch_size, args.gamma, args.tau)
            elif args.policy_name == 'MATD3':
                policy.learn(args.batch_size, args.gamma,args.tau,args.max_action,args.policy_noise,args.noise_clip)
            elif args.policy_name == 'MAAC':
                policy.learn(args.batch_size, args.gamma, args.tau)
            
            #maddpg.update_target(args.tau)

        win_list.append(1 if info["win"] else 0) 
        win1_list.append(1 if info["win1"] else 0)   
        if info["win1"]:
            red_tank_0_hp_l.append(info["Red-tank-0"])
            red_tank_1_hp_l.append(info["Red-tank-1"])
            red_tank_2_hp_l.append(info["Red-tank-2"])

        for agent_id, r in agent_reward.items():  # record reward
            episode_reward[agent_id][episode] = r
            writer.add_scalar(f'agent_{agent_id}_reward', r, episode)
        
        if (episode + 1) % 100 == 0:  # print info every 100 episodes
            message = f'episode {episode + 1}, '
            sum_reward = 0
            for agent_id, r in agent_reward.items():  # record reward
                message += f'{agent_id[4:]}: {r:.1f}; '
                sum_reward += r
            message += f'sum reward: {sum_reward:.1f}; '
            win_rate = np.mean(win_list[-100:]) #一般胜率
            win1_rate = np.mean(win1_list[-100:])



            if args.policy_number == 0: # 只对策略0进行训练 对策略1 进行测试
                if win1_rate >= args.win_rate_adjust :#        
                    policy_up += 1
                    args.noise_std = args.noise_std * 0.9 
                    if args.policy_name == 'MATD3':
                        args.policy_noise = args.policy_noise * 0.9
                        args.noise_clip = args.noise_clip * 0.9
                    if win1_rate >= 0.10 or policy_up > 20:  #之后改成0.05试试
                        args.win_rate_adjust += 0.02
                        episode_temp = episode
                    print(f'策略提升{policy_up}',args.noise_std,args.win_rate_adjust)   
                
                # 动态调整
                if win1_rate >= 0.10 or policy_up > 20: #此时episode_temp才有定义
                    if episode - episode_temp > 10000:
                        args.win_rate_adjust -=0.01
                        episode_temp = episode
                        print(f'策略调整{policy_up}adjust',args.noise_std,args.win_rate_adjust)
            




            writer.add_scalar('win_rate', win_rate, episode)
            writer.add_scalar('win1_rate', win1_rate, episode) #大胜利
            if len(red_tank_0_hp_l) > 100:
                red_tank_0_hp = np.mean(red_tank_0_hp_l[-100:])
                red_tank_1_hp = np.mean(red_tank_1_hp_l[-100:])
                red_tank_2_hp = np.mean(red_tank_2_hp_l[-100:])
                writer.add_scalar('red_tank_0_hp', red_tank_0_hp, episode)
                writer.add_scalar('red_tank_1_hp', red_tank_1_hp, episode)
                writer.add_scalar('red_tank_2_hp', red_tank_2_hp, episode)

            # print message
            message += f'red_hp:'
            for agend_id in env.agents:
                message += f'{info[agend_id]:.1f},'
            message = message[:-1] + ';'
            
            message += f'blue_hp:'
            for agend_id in env.agents_e:
                message += f'{info[agend_id]:.1f},'
            message = message[:-1] + ';'

            message += f'win rate: {win_rate:.2f}; '
            message += f'win1 rate: {win1_rate:.2f}; '
            print(message)
        
        # save model
        if episode == args.max_episodes // 2 :
            name = str(args.policy_number) + '_' +str(episode) 
            policy.save(episode_reward, name)  # save model


        
    # training finishes, plot reward
    name = str(args.policy_number) + '_' +str(args.max_episodes)      
    policy.save(episode_reward, name)  # save model
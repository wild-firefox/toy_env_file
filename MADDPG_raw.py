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
from env_restruction import ToyEnv
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
        # 尝试改4 效果好
        next_action = {}
        for agent_id, buffer in self.buffers.items():
            obs[agent_id], action[agent_id], reward[agent_id], next_obs[agent_id], done[agent_id] = buffer.sample(indices)
            next_action[agent_id] = self.agents[agent_id].actor_target(next_obs[agent_id]) # 区别3

        return obs, action, reward, next_obs, done ,next_action#包含所有智能体的数据
    
    ## DDPG算法相关
    def learn(self, batch_size ,gamma , tau):
        # 多智能体特有-- 集中式训练critic:计算next_q值时,要用到所有智能体next状态和动作
        for agent_id, agent in self.agents.items():
            ## 更新前准备
            obs, action, reward, next_obs, done , next_action= self.sample(batch_size) # 必须放for里，否则报二次传播错，原因是原来的数据在计算图中已经被释放了
            ''' 区别1 '''
            # next_action = {}
            # for agent_id_, agent_ in self.agents.items():
            #     next_action_i = agent_.actor_target(next_obs[agent_id_]) 
            #     next_action[agent_id_] = next_action_i
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
        data = torch.load(os.path.join(model_dir, 'MADDPG.pth'))
        for agent_id, agent in policy.agents.items():
            agent.actor.load_state_dict(data[agent_id])
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
13: 未使用done_bool
14: done_bool
15 减去可视
16 MADDPG 上256 128
17 MADDPG_ 上256 128
18 MADDPG_ 128 128  400000
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
    parser.add_argument("--max_episodes", type=int, default=int(400000))  # 得改成 200000 
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
    episode_reward = {agent_id: 0 for agent_id in env_agents}
    train_return = {agent_id: np.zeros(args.max_episodes) for agent_id in env.agents} 
    obs,info = env.reset(args.policy_number)
    obs = env.Normalization(obs) #状态归一

    while episode_num < args.max_episodes:
        step +=1    

        # 获取动作
        if step < args.random_steps: # # 区分环境里的action_ 和训练的action
            action = env.sample()  # 尝试改2 #{agent: env.action_space(agent).sample() for agent in env_agents}  # [-1,1]
        else:
            action = policy.select_action(obs) # [-1,1]
        # 加噪音 [-1,1] -> [-max_action,max_action]
        action_ = {agent_id: np.clip(action[agent_id] * max_action + np.random.normal(scale = args.gauss_sigma * max_action, size = dim_info[agent_id][1]), -max_action, max_action) for agent_id in env_agents} 
        #print(action_)
        # 探索环境
        next_obs, reward,terminated, truncated, infos = env.step(action_) 
        next_obs = env.Normalization(next_obs) #状态归一

        done = {agent_id: terminated[agent_id] or truncated[agent_id] for agent_id in env_agents}
        done_bool = {agent_id: terminated[agent_id] for agent_id in env_agents} ### truncated 为超过最大步数
        
        if any(done.values()):
            win_list.append(1 if infos["win"] else 0) 
            win1_list.append(1 if infos["win1"] else 0)
            #结算奖励
            for agent_id, r in reward.items():
                if infos['win1'] == True: #只有结算时有 大win
                    reward[agent_id] +=  10  #3*r  #100
                    if infos[agent_id] > 1e-3: 
                        reward[agent_id] +=  3  #存活奖励
                        reward[agent_id] +=  infos[agent_id]*3 #生命值奖励
        
        policy.add(obs, action, reward, next_obs, done_bool) # 尝试改2.2  ## 为什么使用done_bool效果下降了
        for agent_id,r in reward.items():
            episode_reward[agent_id] += r
        #episode_reward = {agent_id: episode_reward[agent_id] + r for agent_id,r in reward.items()} # 尝试改3
        obs = next_obs
        
        # episode 结束 
        if any(done.values()):
            ## 显示
            if  (episode_num + 1) % 100 == 0:
                message = f'episode {episode_num + 1}, '
                #print("episode: {}, reward: {}".format(episode_num + 1, episode_reward))
                win_rate = np.mean(win_list[-100:]) #一般胜率
                win1_rate = np.mean(win1_list[-100:])
                if win1_rate >= args.win_rate_adjust :#        
                    policy_up += 1
                    args.gauss_sigma = args.gauss_sigma * 0.9 
                    if win1_rate >= 0.10 or policy_up > 20:  #之后改成0.05试试
                        args.win_rate_adjust += 0.02
                        episode_temp = episode_num
                    print(f'策略提升{policy_up}',args.gauss_sigma,args.win_rate_adjust)   
                # 动态调整
                if win1_rate >= 0.10 or policy_up > 20: #此时episode_temp才有定义
                    if episode_num - episode_temp > 10000:
                        args.win_rate_adjust -=0.01
                        # 新增
                        args.gauss_sigma = args.gauss_sigma * 0.9 
                        episode_temp = episode_num
                        print(f'策略调整{policy_up}adjust',args.gauss_sigma,args.win_rate_adjust)
                writer.add_scalar('win_rate', win_rate, episode_num)
                writer.add_scalar('win1_rate', win1_rate, episode_num) #大胜利
                # print message
                message += f'red_hp:'
                for agend_id in env.agents:
                    message += f'{infos[agend_id]:.1f},'
                message = message[:-1] + ';'
                
                message += f'blue_hp:'
                for agend_id in env.agents_e:
                    message += f'{infos[agend_id]:.1f},'
                message = message[:-1] + ';'

                message += f'win rate: {win_rate:.2f}; '
                message += f'win1 rate: {win1_rate:.2f}; '
                print(message)

            for agent_id , r in episode_reward.items(): #尝试改3.1
                writer.add_scalar(f'reward_{agent_id}', r, episode_num + 1)
                train_return[agent_id][episode_num] = r  #.append(episode_reward[agent_id])

            episode_num += 1
            obs,info = env.reset(args.policy_number)
            obs = env.Normalization(obs) #状态归一
            episode_reward = {agent_id: 0 for agent_id in env_agents}
        
        # 满足step,更新网络
        #if step > args.start_steps and step % args.learn_steps_interval == 0:
            # 尝试改1
            if step > args.start_steps and episode_num % args.learn_interval == 0:
                policy.learn(args.batch_size, args.gamma, args.tau)

    print('total_time:',time.time()-time_)
    policy.save(model_dir)
    ## 保存数据
    train_return_ = np.array([train_return[agent_id] for agent_id in env.agents])
    if args.N is None:
        np.save(os.path.join(model_dir,f"{args.policy_name}_seed_{args.seed}.npy"),train_return_)
    else:
        np.save(os.path.join(model_dir,f"{args.policy_name}_seed_{args.seed}_N_{len(env_agents)}.npy"),train_return_)
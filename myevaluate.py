import argparse
import os

import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置中文字体为SimHei
plt.rcParams['axes.unicode_minus'] = False  # 正确显示负号
from PIL import Image

from MADDPG_simple import MADDPG
from MADDPG_simple import get_env

#from MAPPO_decay_std import MAPPO
from MAPPO import MAPPO

import math

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', type=str, default='toy_env', help='name of the env',
                        choices=['toy_env'])
    parser.add_argument('--folder', type=str, default='MAPPO_6',help='name of the folder where model is saved')
    parser.add_argument('--episode-num', type=int, default=100, help='total episode num during evaluation')
    parser.add_argument('--episode-length', type=int, default=30, help='steps per episode')
    parser.add_argument('--action-bound', type=float, default=2*math.pi, help='upper bound of action')
    parser.add_argument('--render_mode', type=bool, default=False, help='std of noise')
    parser.add_argument('--gauss_sigma', type=float, default=0.0*2*math.pi, help='std of noise')
    parser.add_argument('--model_name', type=str, default='model_0_200000', help='name of the model to be loaded')
    parser.add_argument('--policy_number', type=int, default=1, help='number of policy')
    parser.add_argument('--policy_noise_a', type=float, default=0.1*2*math.pi, help='policy noise')
    parser.add_argument('--policy_noise_b', type=float, default=0.1*2*math.pi, help='policy noise')

    # PPO tricks
    parser.add_argument("--trick", type=dict, default={'adv_norm':False,
                                                    'ObsNorm':False,
                                                    'reward_norm':False,'reward_scaling':False,    # or
                                                    'orthogonal_init':False,'adam_eps':False,'lr_decay':False, # 原代码中设置为False
                                                    # 以上均在PPO_with_tricks.py中实现过
                                                    'ValueClip':False,'huber_loss':False,
                                                    'LayerNorm':False,'feature_norm':False,
                                                    })  

    args = parser.parse_args()

    # # MAPPO
    if  'MAPPO_decay_std_23'.split('_')[0] == 'MAPPO' or ((args.trick['lr_decay'] is False ) and all(value  for key, value in args.trick.items() if key not in ['reward_norm','lr_decay'])) :
        #args.policy_name = 'MAPPO'
        for key in args.trick.keys():
            if key not in ['reward_norm','lr_decay']:
                args.trick[key] = True
            else:
                args.trick[key] = False

    model_dir = os.path.join('./results', args.env_name, args.folder)
    assert os.path.exists(model_dir)

    gif_dir = os.path.join(model_dir, 'gif')
    if not os.path.exists(gif_dir):
        os.makedirs(gif_dir)
    gif_num = len([file for file in os.listdir(gif_dir)])  # current number of gif

    env,dim_info,max_action,is_continue = get_env(args.env_name,render_mode=False)
    env_agents = [agent_id for agent_id in env.agents]

    policy = MAPPO.load(dim_info, is_continue, model_dir,trick=args.trick)

    agent_num = env.num_agents
    # reward of each episode of each agent
    episode_rewards = {agent: np.zeros(args.episode_num) for agent in env.agents}
    win_list = []
    lose_list = []
    win_rate = []
    lose_rate = []
    win1_list = []
    win1_rate = []
    #env, dim_info = get_env(args.env_name,render_mode=True)# ##改4
    for episode in range(args.episode_num):
        obs,infos = env.reset(args.policy_number,args.policy_noise_a,args.policy_noise_b) ##改1
        obs = env.Normalization(obs) #状态归一
        agent_reward = {agent: 0 for agent in env.agents}  # agent reward of the current episode
        frame_list = []  # used to save gif
        done = {agent_id: False for agent_id in env.agents}
        while not any(done.values()):
            #env.render()
        #while env.agents:  # interact with the env for an episode
            action_nor = policy.evaluate_action(obs)  # 注 混合博弈需要噪声 例:石头剪刀布
            # action = {agent_id: [np.clip(a[0]*args.action_bound   + np.random.normal(0, args.noise_std,), 
            #     -args.action_bound, args.action_bound)]
            #     for agent_id, a in action_nor.items()}
            action_ = {agent_id: np.clip(action_nor[agent_id] * max_action + np.random.normal(scale = args.gauss_sigma * max_action, size = dim_info[agent_id][1]), -max_action, max_action) for agent_id in env_agents} 
            next_obs, reward, terminations, truncations, info = env.step(action_)
            next_obs = env.Normalization(next_obs) #状态归一
            done ={agent_id: terminations[agent_id] or truncations[agent_id] for agent_id in env.agents}
            #frame_list.append(Image.fromarray(env.render())) ## 改3
            obs = next_obs
            
            
            #每个时间步惩罚和结算奖励  
            for agent_id, r in reward.items():
                #reward[agent_id] -= 0.01 # 暂时取消
                #r = abs(reward[agent_id])
                if info['win1'] == True: #只有结算时有 大win
                    reward[agent_id] +=  100#3*r  #100
                elif info['win2'] == True:  #小win
                    reward[agent_id] += 30#r #30

            for agent_id, r in reward.items():  # update reward
                agent_reward[agent_id] += r

        #env.close()
        message = f'episode {episode + 1}, '
        # episode finishes, record reward
        ### 记录胜率曲线
        win_list.append(1 if info["win"] else 0)
        lose_list.append(1 if info["lose"] else 0)
        win1_list.append(1 if info["win1"] else 0)
        win_rate.append(sum(win_list)/(episode+1))
        lose_rate.append(sum(lose_list)/(episode+1))
        win1_rate.append(sum(win1_list)/(episode+1))
        for agent_id, reward in agent_reward.items():
            episode_rewards[agent_id][episode] = reward
            message += f'{agent_id}: {reward:.2f}; '
        message += f'win_rate: {sum(win_list)/(episode+1):.3f}; '
        message += f'lose_rate: {sum(lose_list)/(episode+1):.3f}; '
        message += f'win1_rate: {sum(win1_list)/(episode+1):.3f}; '
        print(message)
        # save gif
        # frame_list[0].save(os.path.join(gif_dir, f'out{gif_num + episode + 1}.gif'),
        #                    save_all=True, append_images=frame_list[1:], duration=1, loop=0)



    #创建test文件夹
    test_dir = os.path.join(model_dir, 'test')
    if not os.path.exists(test_dir):
        os.makedirs(test_dir)

    # ## 读取大胜率 numpy 绘图
    # win1_rate = np.load(os.path.join(model_dir, 'MADDPG_simple_seed_0_win1.npy'))
    
    # fig, ax = plt.subplots()
    # x = range(1, args.episode_num + 1)
    # ax.plot(x, win1_rate, label='大胜率')
    # ax.legend()
    # ax.set_xlabel('片段数')
    # ax.set_ylabel('大胜率')
    # title = f'win1_rate of maddpg vs {args.policy_number} in {args.env_name}'
    # plt.savefig(os.path.join(test_dir, title))

    # ## 读取胜率 numpy 绘图
    # win_rate = np.load(os.path.join(model_dir, 'MADDPG_simple_seed_0_win.npy'))

    # fig, ax = plt.subplots()
    # x = range(1, args.episode_num + 1)
    # ax.plot(x, win_rate, label='胜率')
    # ax.legend()
    # ax.set_xlabel('片段数')
    # ax.set_ylabel('胜率')
    # title = f'win_rate of maddpg vs {args.policy_number} in {args.env_name}'
    # plt.savefig(os.path.join(test_dir, title))



    
    
    
    total_files = len([file for file in os.listdir(test_dir)]) // 3 + 1
    #total_files = total_files-1 if os.path.exists(gif_dir) else total_files
    #total_files = (total_files-1) // 2 - 1

    # training finishes, plot reward
    fig, ax = plt.subplots()
    x = range(1, args.episode_num + 1)
    for agent_id, rewards in episode_rewards.items():
        ax.plot(x, rewards, label='己方'+agent_id[-1])
    ax.legend()
    ax.set_xlabel('片段数')
    ax.set_ylabel('奖励值')
    title = f'reward function of evaluate result of maddpg solve {args.env_name} ( vs {args.policy_number})'
    title_zn = f'奖励函数曲线'
    ax.set_title(title_zn)
    title += f'{total_files}'
    plt.savefig(os.path.join(test_dir, title))

    fig, ax = plt.subplots()
    x = range(1, args.episode_num + 1)
    ax.plot(x, win_rate, label='胜率')
    ax.plot(x, lose_rate, label='负率')
    ax.legend()
    ax.set_xlabel('片段数')
    ax.set_ylabel('胜负率')
    title = f'win_lose_rate of maddpg solve {args.env_name} ( vs {args.policy_number})'
    title_zn = f'胜负率曲线'
    ax.set_title(title_zn)
    title += f'{total_files}'
    plt.savefig(os.path.join(test_dir, title))

    # plot win1 rate
    fig, ax = plt.subplots()
    x = range(1, args.episode_num + 1)
    ax.plot(x, win1_rate, label='大胜率')
    ax.legend()
    ax.set_xlabel('片段数')
    ax.set_ylabel('大胜率')
    title = f'win1_rate of maddpg solve {args.env_name} ( vs {args.policy_number})'
    title_zn = f'大胜率曲线'
    ax.set_title(title_zn)
    title += f'{total_files}'
    plt.savefig(os.path.join(test_dir, title))

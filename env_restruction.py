"""3v3坦克对抗环境实现"""
import math
import numpy as np

## 数据类型
from collections import deque
from collections import defaultdict
from mytyping import Action, Info, Number, Observation, Reward, TeamName
from gymnasium import spaces

## 配置
from pathlib import Path
import addict  # 允许通过属性访问字典的键
import toml # 读取后为嵌套字典
import os
'''
pip install addict
pip install toml
'''

# render
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置中文字体为SimHei
plt.rcParams['axes.unicode_minus'] = False  # 正确显示负号
from matplotlib.patches import Circle
import time

'''' 此为 env的重建版 去掉一些不必要的注释 并优化代码
现仅保留 3坦克 vs 3坦克的对抗 
一些未加入的全注释掉：（在env.py里有）
    1.士兵 加入后 未能学到好的策略 删去
    2.可视 加入后收敛变慢 删去
'''



# from env_restruction import * 时导入的内容
__all__ = ["ToyEnv", "config"]

# 读取配置 得到状态范围
config = addict.Dict(toml.load(Path(__file__).parent / "config.toml"))
config.observation_space = {
            # 共用
            "x": [0, config.map.width],
            "y": [0, config.map.height],

            # 坦克
            "hp_t": [0, config.tank.hp],
            "speed_t": [0, config.tank.speed],
            "bullet_t": [0, config.tank.bullet_num], 
            "fuel": [0, config.tank.fuel],

        }


class Tank:
    """坦克类实现。

    :param x: 坦克的起始横坐标
    :type x: float
    :param y: 坦克的起始纵坐标
    :type y: float
    """

    id_counter: dict[str, int] = {"Red": 0, "Blue": 0}

    def __init__(self, x: float, y: float, team: TeamName) -> None:
        assert team in TeamName.__args__
        self.x = x
        self.y = y
        self._speed = config.tank.speed
        self.speed = 0
        self.team = team
        self.x_max = config.map.width
        self.y_max = config.map.height
        self.hp = config.tank.hp
        self.fuel = config.tank.fuel
        #-- 新增可视范围
        self.visibility_range = config.tank.visibility_range
    
        self.bullets = deque(
            [
                Bullet(config.missile.damage_radius, self)
                for _ in range(config.tank.bullet_num)
            ],
        )
        self.consumed_bullets = deque[Bullet]()
        self.shoot_distance = config.tank.shoot_distance
        self.uid = f"{team}-tank-{Tank.id_counter[team]}"
        Tank.id_counter[team] += 1

    @staticmethod
    def clip(v: Number, _min: Number, _max: Number) -> Number:
        """将数值限制在上下界之内。"""
        if v > _max:
            return _max
        elif v < _min:
            return _min
        else:
            return v

    @property #像属性一样访问,可以直接调用不用加括号
    def alive(self) -> bool:
        """指示坦克是否存活。"""
        return self.hp > 1e-3 #1e-3是表示接近一定程度就算死亡

    @property
    def movable(self) -> bool:
        """指示坦克是否还能移动。"""
        return self.alive and self.fuel > 0

    @property
    def shootable(self) -> bool:
        """指示坦克是否还能射击。"""
        return self.alive and len(self.bullets) > 0

    def move(self, angle: float) -> None:
        r"""指定角度移动。

        :param angle: 朝哪个方向移动（弧度制 :math:`0\le angle\le 2\pi`）
        """
        assert 0 <= angle <= 2 * math.pi #+ 1e-3
        if abs(angle - 2 * math.pi) < 1e-3: #这里1e-3是表示接近一定程度就取到边界值
            angle = 2 * math.pi
        if abs(angle - 0) < 1e-3:
            angle = 0

        if self.movable is False:
            self.speed = 0
            return
        self.speed = self._speed
        cos = math.cos(angle)
        sin = math.sin(angle)
        if abs(cos) < 1e-7: #这里的 防止浮点数误差
            cos = 0
        if abs(sin) < 1e-7:
            sin = 0
        delta_x = self.speed * cos
        delta_y = self.speed * sin 

        self.x = Tank.clip(self.x + delta_x, 0, self.x_max)
        self.y = Tank.clip(self.y + delta_y, 0, self.y_max)
            
        for b in self.bullets:
            b.x = self.x
            b.y = self.y
        #dis = math.sqrt(delta_x ** 2 + delta_y ** 2)
        self.fuel -= 1  ### 一次动5distance
        self.fuel = max(0, self.fuel)

    def shoot(self, angle: float) -> None:
        r"""朝指定角度射击。
        :param angle: 朝哪个方向射击（弧度制 :math:`0\le angle\le 2\pi`）
        这里射击距离固定是shoot_distance
        """
        assert 0 <= angle <= 2 * math.pi #+ 1e-3
        if self.shootable is False:
            return
        if abs(angle - 2 * math.pi) < 1e-3: #这里1e-3是表示接近一定程度就取到边界值
            angle = 2 * math.pi
        if abs(angle - 0) < 1e-3:
            angle = 0
        
        self.speed = 0
        b = self.bullets.pop()
        cos = math.cos(angle)
        sin = math.sin(angle)
        if abs(cos) < 1e-7:
            cos = 0
        if abs(sin) < 1e-7:
            sin = 0
        b.x += self.shoot_distance *  cos
        b.y += self.shoot_distance * sin
        b.x = Tank.clip(b.x, 0, self.x_max)
        b.y = Tank.clip(b.y, 0, self.y_max)
        #print(self.uid,b.x,b.y)
        self.consumed_bullets.append(b)


class Bullet:
    r"""子弹类实现。

    :param radius: 杀伤半径
    :param owner: 子弹的所有者
    :param damage: 中心伤害
    """

    def __init__(self, radius: float, owner: Tank, damage: float = 3) -> None:
        self.radius = radius
        self.owner = owner
        self.x = owner.x
        self.y = owner.y
        self.damage = damage

class Soldier:
    """步兵类实现。

    :param x: 步兵的起始横坐标
    :type x: float
    :param y: 步兵的起始纵坐标
    :type y: float

    士兵特点:无燃油值 除非死,否则可以一直移动
    """

    id_counter: dict[str, int] = {"Red": 0, "Blue": 0}

    def __init__(self, x: float, y: float, team: TeamName) -> None:
        assert team in TeamName.__args__
        self.x = x
        self.y = y
        self._speed = config.soldier.speed
        self.speed = 0
        self.team = team
        self.x_max = config.map.width
        self.y_max = config.map.height
        self.hp = config.soldier.hp
        
        #-- 新增可视范围
        self.visibility_range = config.soldier.visibility_range
        self.bullets = deque(
            [
                Bullet_s(self, config.bullet.damage)
                for _ in range(config.soldier.bullet_num)
            ],
        )
        self.consumed_bullets = deque[Bullet_s]()
        self.shoot_distance = config.soldier.shoot_distance
        self.uid = f"{team}-soldier-{Soldier.id_counter[team]}"
        Soldier.id_counter[team] += 1

    @staticmethod
    def clip(v: Number, _min: Number, _max: Number) -> Number:
        """将数值限制在上下界之内。"""
        if v > _max:
            return _max
        elif v < _min:
            return _min
        else:
            return v

    @property
    def alive(self) -> bool:
        """指示步兵是否存活。"""
        return self.hp > 1e-3

    @property
    def shootable(self) -> bool:
        """指示步兵是否还能射击。"""
        return self.alive and len(self.bullets) > 0
    
    @property
    def movable(self) -> bool:
        """指示坦克是否还能移动。"""
        return self.alive  

    def move(self, angle: float) -> None:
        r"""指定角度移动。

        :param angle: 朝哪个方向移动（弧度制 :math:`0\le angle\le 2\pi`）
        """
        assert 0 <= angle <= 2 * math.pi
        if abs(angle - 2 * math.pi) < 1e-3: #这里1e-3是表示接近一定程度就取到边界值
            angle = 2 * math.pi
        if abs(angle - 0) < 1e-3:
            angle = 0

        if self.movable is False:
            self.speed = 0
            return
        self.speed = self._speed

        cos = math.cos(angle)
        sin = math.sin(angle)
        if abs(cos) < 1e-7: #这里的 防止浮点数误差
            cos = 0
        if abs(sin) < 1e-7:
            sin = 0
        delta_x = self.speed * cos 
        delta_y = self.speed * sin 

        self.x = Soldier.clip(self.x + delta_x, 0, self.x_max)
        self.y = Soldier.clip(self.y + delta_y, 0, self.y_max)
        for b in self.bullets:
            b.x = self.x
            b.y = self.y

    def shoot(self, angle: float) -> None:
        r"""朝指定角度射击。

        :param angle: 朝哪个方向射击（弧度制 :math:`0\le angle\le 2\pi`）
        """
        assert 0 <= angle <= 2 * math.pi + 1e-3
        if abs(angle - 2 * math.pi) < 1e-3: #这里1e-3是表示接近一定程度就取到边界值
            angle = 2 * math.pi
        if abs(angle - 0) < 1e-3:
            angle = 0
            
        if self.shootable is False:
            return
        self.speed = 0
        b = self.bullets.pop()
        cos = math.cos(angle)
        sin = math.sin(angle)
        if abs(cos) < 1e-7:
            cos = 0
        if abs(sin) < 1e-7:
            sin = 0
        b.x += self.shoot_distance *  cos
        b.y += self.shoot_distance * sin

        b.x = Soldier.clip(b.x, 0, self.x_max)
        b.y = Soldier.clip(b.y, 0, self.y_max)
        self.consumed_bullets.append(b)


class Bullet_s:
    r"""子弹类实现。

    :param owner: 导弹的所有者
    :param damage: 中心伤害
    """

    def __init__(self, owner: Soldier, damage: float) -> None:
        self.owner = owner
        self.x = owner.x
        self.y = owner.y
        self.damage = damage

class Team:
    r"""队伍类实现，一共可能有两种类型：
    * 红队
    * 蓝队
    :param name: 队伍名称
    """

    def __init__(self, name: TeamName) -> None:
        assert name in TeamName.__args__
        self.name = name
        _y = config.map.height if self.name == "Blue" else 0
        
        ## 坦克 （x,y name）
        self.tanks = [
            Tank(
                config.map.width / 2
                + config.map.width / 5 * ((i + 1) // 2) * (-1) ** i,
                _y,
                self.name,
            )
            for i in range(config.team.get("n_{}_tank".format(self.name.lower())))
        ]
        
        ## 士兵
        self.soldiers = [
            Soldier(
                config.map.width / 2
                + config.map.width / 20 * ((i + 1) // 2) * (-1) ** i,
                _y,
                self.name,
            )
            for i in range(config.team.get("n_{}_soldier".format(self.name.lower())))
        ]

    @property
    def alives(self) -> list[bool]:
        """获取队伍所有人员的存活状态"""
        return [t.alive for t in self.tanks+ self.soldiers]

## 环境主体
class ToyEnv:
    """
    简易游戏环境。
    设定：先攻击的一方先结算伤害
    """

    def __init__(self,render_mode=False) -> None:
        self.render_mode = render_mode
        self.step_size = 30 / config.max_steps
        self.red_team = Team("Red")
        self.blue_team = Team("Blue")
        self.step_ = 0  # 记录当前回合数
        self.agents = [i.uid for i in self.red_team.tanks + self.red_team.soldiers]  #要进行训练的智能体
        self.agents_e = [i.uid for i in self.blue_team.tanks + self.blue_team.soldiers] #敌方
        self.episode_length = config.max_steps
        
        if self.render_mode == True:
            print('render_mode:',self.render_mode)
            self.init_render()

    # 环境内置属性
    @property
    def num_agents(self) -> int:
        return len(self.agents)
    
    # 关于坦克的内置属性
    @property
    def tanks(self) -> list[Tank]:
        """所有的坦克实例。"""
        return self.red_team.tanks + self.blue_team.tanks
    @property
    def red_tanks(self) -> list[Tank]:
        """红队的坦克实例。"""
        return self.red_team.tanks
    @property
    def blue_tanks(self) -> list[Tank]:
        """蓝队的坦克实例。"""
        return self.blue_team.tanks
    
    ## 关于士兵的内置属性和用到的方法
    @staticmethod # 与类状态无关但功能相关
    def compute_distance(*args) -> float:
        if len(args) == 2:
            o1, o2 = args
            d = math.sqrt((o1.x - o2.x) ** 2 + (o1.y - o2.y) ** 2)
        else:
            o1, o2, o3 = args
            a = o2.y - o1.y
            b = o1.x - o2.x
            c = o2.x * o1.y - o1.x * o2.y
            d = (a * o3.x + b * o3.y + c) / math.sqrt((a**2 + b**2 + 1e-3))
        return d

    @staticmethod
    def atan(o1, o2) -> float:
        theta = math.atan2(o2.y - o1.y, o2.x - o1.x)
        if theta < 0:
            theta += 2 * math.pi
        return theta
    
    @property
    def soldiers(self) -> list[Soldier]:
        """所有的士兵实例。"""
        return self.blue_team.soldiers + self.red_team.soldiers

    @property
    def red_soldiers(self) -> list[Soldier]:
        """红队的士兵实例。"""
        return self.red_team.soldiers

    @property
    def blue_soldiers(self) -> list[Soldier]:
        """蓝队的士兵实例。"""
        return self.blue_team.soldiers

    # 环境的内置属性
    def sample(self) -> dict[str, Action]:
        """随机生成动作,后续在环境采样时* max_action """  
        actions = {}
        for t in self.red_tanks + self.red_soldiers:
            #生成-1-1之间的随机数
            actions[t.uid] = np.array([ np.random.uniform(-1, 1) ])
        return actions
    
    def blue_policy(self,num:int,noise_a_,noise_b_): 
        """
        0: 敌方（蓝方）随机生成动作并执行: 设定：不对己方自己及队友造成伤害(完成) -- 使用随机策略训练出的智能体拓展性好   （当训练集）
        1：根据当前状态选择动作 :攻击在范围内最近的“活着的”敌人，如果攻击范围内没有敌人，则向敌人靠近,若离敌人太近则后退。  （当测试集）
        设定：会有己伤 ，有两个超参数来调整敌人的强度：noise_a_ 前进的高斯噪音 默认0.12*2*math.pi ,noise_b_ 后退和射击的噪音0.1*2*math.pi (未完善:加入士兵)
        2:不动策略
        3:随机选择策略0或者策略1、2
        """
        actions = {}
        match num:
            case 0:
                for t in self.blue_tanks + self.blue_soldiers:
                    #生成-1-1之间的随机数
                    actions[t.uid] = [ np.random.uniform(-1, 1) * 2 * math.pi]
                
                self.assign_actions(actions)
                
                ### 结算伤害 --坦克
                for  t in (self.blue_tanks):  #
                    if t.alive is False:
                        continue
                    for b in t.consumed_bullets: #蓝方导弹

                        for tt in self.red_tanks + self.red_soldiers:  #改成不对自己造成伤害tanks->red_tanks
                            if tt.alive is False:
                                continue
                            distance = math.sqrt((b.x - tt.x) ** 2 + (b.y - tt.y) ** 2)
                            if distance < b.radius:
                                damage = (1 - distance**2 / (b.radius**2)) * b.damage
                                damage = min(damage, tt.hp)
                                tt.hp -= damage
                                tt.hp = 0 if tt.hp <= 1e-3 else tt.hp
                                if isinstance(tt, Tank):
                                    if distance < (b.radius/2):
                                        tt.fuel -= min(10//self.step_size,tt.fuel)
                                        tt.fuel = max(0, tt.fuel)

                ## 结算伤害 士兵
                for  s in (self.blue_soldiers):
                    if s.alive is False:
                        continue
                    for b in s.consumed_bullets:
                        for ss in self.red_soldiers:  # 只对士兵造成伤害
                            if b.owner.uid == ss.uid: #自己的子弹不对自己造成伤害
                                continue
                            if ss.alive is False: #如果已经死亡就不用计算了
                                continue
                            if self.compute_distance(s, ss) > s.shoot_distance: #不在射击范围内
                                continue
                            theta1 = self.atan(s, b) # 士兵与子弹的夹角
                            theta2 = self.atan(s, ss) # 士兵与士兵的夹角
                            theta = theta1 - theta2 # 士兵与子弹的夹角与士兵与士兵的夹角之差
                            if theta < 0:
                                theta += 2 * math.pi
                            if math.pi < theta < 2 * math.pi: # 180-360度跳过
                                continue
                            d = self.compute_distance(s, b, ss) # 士兵与子弹轨迹的垂直距离
                            if d < 2:
                                damage = b.damage
                                ss.hp -= damage
                                ss.hp = 0 if ss.hp <= 1e-3 else ss.hp

            case 1: #暂未加入士兵 (已加入)
                
                # 坦克对坦克策略    之后改 先打坦克再打士兵
                for t in self.blue_tanks:
                    # 如上一个打死 则下一个
                    red_xy = [[t.x, t.y] for t in self.red_tanks if t.alive is True]  ##bug修复 死了一个后索引错误
                    if len(red_xy) == 0:
                        continue
                    dis = [math.sqrt((t.x - red_xy[i][0])**2 + (t.y - red_xy[i][1])**2) for i in range(len(red_xy))] #红方都死的话 不会进行到这步
                    min_dis = min(dis)
                    min_index = dis.index(min_dis)
                    angle = math.atan2(red_xy[min_index][1] - t.y, red_xy[min_index][0] - t.x) #-pi-pi
                    #angle = (angle + 2 * math.pi) % (2 * math.pi) #0-2pi # 与下两行一致
                    if angle < 0:
                        angle += 2 * math.pi # 0-2pi
                    
                    attack_angle = angle
                    back_angle = angle + math.pi #1pi-3pi
                    # 保证angle在0-2pi之间
                    if back_angle > 2 * math.pi:
                        back_angle -= 2 * math.pi
                    
                    # 加噪音 
                    noise_a =  np.random.normal(0, noise_a_) #返回高斯分布
                    noise_b =  np.random.normal(0, noise_b_)

                    angle = np.clip(angle + noise_a , 0, 2*math.pi) #前进
                    back_angle = np.clip(back_angle + noise_b, 0, 2*math.pi) #后退 noise与射击一样
                    attack_angle = np.clip(attack_angle + noise_b , 0, 2*math.pi) #射击

                    # 之后加 若无弹药 则后退
                    if 10 < min_dis < 20 : #如果在射击范围内 15+5 在等于20的时候无伤害
                        actions[t.uid] = [-attack_angle]
                
                    elif 20 <= min_dis : #不在射击范围内,向敌人靠近
                        actions[t.uid] = [angle]
                    
                    elif min_dis < 10: #离太近,则后退
                        actions[t.uid] = [back_angle]

                #print(actions)

                ## 加入士兵策略 士兵对士兵策略 士兵进攻最近的敌方士兵
                for s in self.blue_soldiers:
                    red_xy = [[s.x, s.y] for s in self.red_soldiers if s.alive is True]
                    if len(red_xy) == 0:
                        continue
                    dis = [math.sqrt((s.x - red_xy[i][0])**2 + (s.y - red_xy[i][1])**2) for i in range(len(red_xy))]
                    min_dis = min(dis)
                    min_index = dis.index(min_dis)
                    angle = math.atan2(red_xy[min_index][1] - s.y, red_xy[min_index][0] - s.x) #-pi-pi
                    if angle < 0:
                        angle += 2 * math.pi
                    attack_angle = angle
                    back_angle = angle + math.pi #1pi-3pi
                    # 保证angle在0-2pi之间
                    if back_angle > 2 * math.pi:
                        back_angle -= 2 * math.pi
                    # 加噪音
                    noise_a =  np.random.normal(0, noise_a_)
                    noise_b =  np.random.normal(0, noise_b_)
                    angle = np.clip(angle + noise_a , 0, 2*math.pi) #前进
                    back_angle = np.clip(back_angle + noise_b, 0, 2*math.pi) #后退 后两一样
                    attack_angle = np.clip(attack_angle + noise_b , 0, 2*math.pi) #射击
                    # 之后加 若无弹药 则后退
                    if 8 <= min_dis <= 10 : 
                        actions[s.uid] = [-attack_angle]
                    
                    elif 10 < min_dis : #不在射击范围内,向敌人靠近
                        actions[s.uid] = [angle]
                    
                    elif min_dis < 8: #离太近,则后退
                        actions[s.uid] = [back_angle]
                
                
                self.assign_actions(actions)
                ### 结算伤害 --坦克
                for  t in (self.blue_tanks ):  #
                    if t.alive is False:
                        continue
                    for b in t.consumed_bullets: #蓝方导弹
                        for tt in self.tanks + self.soldiers:  
                            if tt.alive is False:
                                continue
                            distance = math.sqrt((b.x - tt.x) ** 2 + (b.y - tt.y) ** 2)
                            if distance < b.radius:
                                damage = (1 - distance**2 / (b.radius**2)) * b.damage
                                tt.hp -= min(damage, tt.hp)
                                tt.hp = 0 if tt.hp <= 1e-3 else tt.hp
                                if isinstance(tt, Tank):
                                    if distance < (b.radius/2):
                                        tt.fuel -= min(10//self.step_size,tt.fuel)  ###增加燃油值功能
                                        tt.fuel = max(0, tt.fuel)
                ### 结算伤害 --士兵
                for  s in (self.blue_soldiers):
                    if s.alive is False:
                        continue
                    for b in s.consumed_bullets:
                        for ss in self.soldiers:
                            if b.owner.uid == ss.uid:
                                continue
                            if ss.alive is False:
                                continue
                            if self.compute_distance(s, ss) > s.shoot_distance:
                                continue
                            theta1 = self.atan(s, b)
                            theta2 = self.atan(s, ss)
                            theta = theta1 - theta2
                            if theta < 0:
                                theta += 2 * math.pi
                            if math.pi < theta < 2 * math.pi:
                                continue
                            d = self.compute_distance(s, b, ss)
                            if d < 2:
                                damage = b.damage
                                ss.hp -= damage
                                ss.hp = 0 if ss.hp <= 1e-3 else ss.hp

            case 2: #蓝方不动策略
                pass

            case 3:
                # 随机选择case0 或者 case1
                num = np.random.randint(3)
                if num == 0:
                    self.blue_policy(0)
                elif num == 1:
                    self.blue_policy(1)
                elif num == 2:
                    self.blue_policy(2)


    def assign_actions(self, actions: dict[str, Action]) -> None:
        """按照uid匹配并执行动作。
        :param actions: 一个字典，用来表征坦克的动作，键是坦克的uid，值是动作，由动作类型和动作值两部分组成
        """
        for uid, a in actions.items():
            flag = False  
            for t in self.tanks + self.soldiers:
                if uid == t.uid:
                    if t.alive is False:
                        continue
                    if a[0]>0:
                        t.move(a[0])
                        flag = True
                    elif a[0]<0:
                        t.shoot(abs(a[0])) #射击时就是不动的
                        flag = True
                    elif a[0] == 0:
                        t.speed = 0
                        flag = True
                if flag is True:
                    break ##匹配到就跳出

    def get_reward_t(self) :
        r"""扫描当前环境，使伤害生效，并提取奖励。\
        伤害计算公式为：

        .. math:: d = [1-(\frac{r}{R})^2]*damage

        其中 :math:`d` 是对某个坦克造成的伤害， :math:`r` 是该坦克离爆炸中心的距离，:math:`R` 是最远伤害范围，超出该范围收到的伤害则为0。

        :return: 一个二元组，分别代表红队和蓝队获得的奖励。
        这里的奖励是奖励到每个个体，在多智能体的设定下 为利己主义
        """
        #三个红方坦克的奖励
        reward = {t.uid : 0.0 for t in self.red_tanks} #defaultdict(int)

        # # 碰壁惩罚
        for t in self.red_tanks:
            if t.x == 0 or t.x == config.map.width or t.y == 0 or t.y == config.map.height:
                reward[t.uid] -= 0.1*self.step_size
        
        # 红方坦克的奖励 和结算伤害
        for  t in (self.red_tanks):
            if t.alive is False:
                continue
            for b in t.consumed_bullets: #红方导弹
                for tt in self.tanks + self.soldiers:  ###加入士兵
                    if tt.alive is False: #如果已经死亡就不用计算了
                        continue
                    distance = math.sqrt((b.x - tt.x) ** 2 + (b.y - tt.y) ** 2)
                    if distance < b.radius:
                        damage = (1 - distance**2 / (b.radius**2)) * b.damage
                        damage = min(damage, tt.hp) #最多扣除hp
                        tt.hp -= damage
                        tt.hp = 0 if tt.hp <= 1e-3 else tt.hp
                        if tt.team == "Blue":
                            reward[t.uid] += damage
                        elif tt.team == "Red":
                            reward[t.uid] -= damage
                        if tt.alive is False:  #当前导弹造成的死 只执行一次
                            if tt.team == "Blue":
                                reward[t.uid] += 3
                            elif tt.team == "Red":
                                reward[t.uid] -= 3
                        ###增加燃油值功能:导弹击中后燃油减少
                        # 判断是否是坦克
                        if isinstance(tt, Tank):
                            if distance < (b.radius/2):
                                tt.fuel -= min(10//self.step_size,tt.fuel) 
                                tt.fuel = max(0, tt.fuel)
                                if tt.team == "Blue":
                                    reward[t.uid] += 1
                                elif tt.team == "Red":
                                    reward[t.uid] -= 1                         

        for t in self.tanks:
            t.consumed_bullets.clear()

        return reward # r1, -r1
    
    def get_reward_s(self) :
        r"""扫描当前环境，使子弹伤害生效，并提取奖励， 最后清除子弹 \
        士兵与子弹轨迹的垂直距离在一定范围d内时，可以狙杀该士兵，否则无法造成伤害：

        .. math:: d = \frac{|Ax_0+By_0+C|}{\sqrt{A^2+B^2}}

        :return: 红方奖励2
        """
        #三个红方士兵的奖励
        reward = reward = {t.uid : 0.0 for t in self.red_soldiers}#defaultdict(int)
        
        # # 碰壁惩罚
        for t in self.red_soldiers:
            if t.x == 0 or t.x == config.map.width or t.y == 0 or t.y == config.map.height:
                reward[t.uid] -= 0.1*self.step_size

        ## 伤害奖励
        for s in self.red_soldiers:
            if t.alive is False:
                continue
            for b in s.consumed_bullets:
                for ss in self.soldiers:  # 只对士兵造成伤害
                    if b.owner.uid == ss.uid: #自己的子弹不对自己造成伤害
                        continue
                    if ss.alive is False: #如果已经死亡就不用计算了
                        continue
                    if self.compute_distance(s, ss) > s.shoot_distance: #不在射击范围内
                        continue
                    theta1 = self.atan(s, b) # 士兵与子弹的夹角
                    theta2 = self.atan(s, ss) # 士兵与士兵的夹角
                    theta = theta1 - theta2 # 士兵与子弹的夹角与士兵与士兵的夹角之差
                    if theta < 0:
                        theta += 2 * math.pi
                    if math.pi < theta < 2 * math.pi: # 180-360度跳过
                        continue
                    d = self.compute_distance(s, b, ss) # 士兵与子弹轨迹的垂直距离
                    if d < 2:
                        damage= config.bullet.damage
                        ss.hp -= damage
                        ss.hp = 0 if ss.hp <= 1e-3 else ss.hp
                        if ss.team == "Blue":
                            reward[s.uid] += damage
                        elif ss.team == "Red":
                            reward[s.uid] -= damage
                        if ss.alive is False:
                            if ss.team == "Blue":
                                reward[s.uid] += 1
                            elif ss.team == "Red":
                                reward[s.uid] -= 1
        for s in self.soldiers:
            s.consumed_bullets.clear()

        return reward

    def get_reward(self):
        reward_t = self.get_reward_t()
        reward_s = self.get_reward_s()
        reward = reward_t | reward_s

        return reward

    def get_obs(self) -> Observation:
        """获取环境观测信息，包含每个坦克，无论是否死亡。

        :return: 返回每个坦克的（坐标，生命值，速度，子弹剩余量，燃油剩余量,
        仅当简单环境来使用，没有改成锥形视野和获得队友的其他属性值（如是否存活），且给的位置为绝对位置 而不是相对位置。
        """
        ## 友方位置
        red_t_xy = [np.array([t.x, t.y], dtype=np.float32) for t in self.red_tanks]
        red_s_xy = [np.array([t.x, t.y], dtype=np.float32) for t in self.red_soldiers]
        red_team_xy = red_t_xy + red_s_xy

        ## 敌方位置
        blue_t_xy = [np.array([t.x, t.y], dtype=np.float32) for t in self.blue_tanks] 
        blue_s_xy = [np.array([t.x, t.y], dtype=np.float32) for t in self.blue_soldiers]
        blue_team_xy = blue_t_xy + blue_s_xy

        obs = {}
        # 更新当前坦克的观测信息
        for t in self.red_tanks:
            obs.update(
                {
                    t.uid: np.array([
                        t.x , t.y,
                        t.hp, 
                        t.speed, 
                        len(t.bullets),
                        t.fuel,
                        # 加入可视时 下三行注释掉
                        blue_t_xy[0][0], blue_t_xy[0][1],  
                        blue_t_xy[1][0], blue_t_xy[1][1], 
                        blue_t_xy[2][0], blue_t_xy[2][1],
                    ], dtype=np.float32)
                }
            )
        
        '''取消可视
        ## 测试1 可视判定 + 动态调整人员个数 不可视的人员坐标为[-1,-1] #好像效果还可以      
        ## 友方位置(包含自己)  
        for t in self.red_tanks:
            agent_pos = np.array([t.x ,t.y])
            for i in range(len(red_team_xy)):
                dis = np.linalg.norm(agent_pos - red_team_xy[i])
                if dis > t.visibility_range:
                    red_team_xy[i] = np.array([-1, -1]) 
                obs.update(
                    {
                        t.uid : np.concatenate([obs[t.uid], red_team_xy[i]])
                    }
                )

        ## 敌方位置
        for t in self.red_tanks:
            agent_pos = np.array([t.x ,t.y])
            for i in range(len(blue_team_xy)):
                dis = np.linalg.norm(agent_pos - blue_team_xy[i])
                if dis > t.visibility_range:
                    blue_team_xy[i] = np.array([-1, -1]) 
                obs.update(
                    {
                        t.uid : np.concatenate([obs[t.uid], blue_team_xy[i]])
                    }
                )
            
        # 更新当前士兵的观测信息
        for s in self.red_soldiers:
            obs.update(
                {
                    s.uid: np.array([
                        s.x , s.y,
                        s.hp,
                        s.speed,
                        len(s.bullets),
                    ], dtype=np.float32)
                }
            )
        
        # 加入可视
        for s in self.red_soldiers:
            agent_pos = np.array([s.x ,s.y])
            for i in range(len(red_team_xy)):
                dis = np.linalg.norm(agent_pos - red_team_xy[i])
                if dis > s.visibility_range:
                    red_team_xy[i] = np.array([-1, -1]) 
                obs.update(
                    {
                        s.uid : np.concatenate([obs[s.uid], red_team_xy[i]])
                    }
                )

        for s in self.red_soldiers:
            agent_pos = np.array([s.x ,s.y])
            for i in range(len(blue_team_xy)):
                dis = np.linalg.norm(agent_pos - blue_team_xy[i])
                if dis > s.visibility_range:
                    blue_team_xy[i] = np.array([-1, -1]) 
                obs.update(
                    {
                        s.uid : np.concatenate([obs[s.uid], blue_team_xy[i]])
                    }
                )
        '''
        return obs

    @property
    def terminated(self) -> bool:
        #"""判断对战是否结束，当一方人员全部阵亡（即生命值全部不大于0）时，对战结束，返回True并结束当前回合。"""
        
        # 红方阵亡即游戏结束 或者 蓝方阵亡即游戏结束
        tem = any(self.red_team.alives) is False or any(self.blue_team.alives) is False
        terminated = {}
        for t in self.agents:
            terminated[t] = tem 

        return terminated #any(self.red_team.alives) is False or any(self.blue_team.alives) is False

    @property
    def truncated(self) -> bool:
        """1.判断对战是否超出最大回合长度，超出时返回True并结束当前回合。
           2.当双方都无法造成伤害时，也结束当前回合。
        """
        tr1 = bool(self.step_ >= config.max_steps)
        tr2 = [ t.alive is False or len(t.bullets) == 0  for t in self.tanks+ self.soldiers]
        #print(tr2)
        tr = tr1 or all(tr2) 
        truncated ={}
        for t in self.agents:
            truncated[t] = tr
        return truncated #self.n >= config.max_steps
    

    def get_info(self) -> Info:
        """获取额外信息。测试2时加上
        这里获取是否胜利或失败
        胜利: 1.蓝方全军覆没 2.红方坦克存活且总hp值大于蓝方
        失败: 1.红方全军覆没 2.蓝方坦克存活且总hp值大于等于红方
        """ 
        infos = {'win1': False, 'win2':False,'lose1': False, 'lose2':False,'win': False, 'lose': False}
        tem = any(self.terminated.values())
        tru = any(self.truncated.values())
        done = tem or tru #修复bug 没给双方无法造成伤害时奖励的bug
        if done:
            if tem: #如果是因为阵亡结束的
                if any(self.blue_team.alives) is False: #win1 win2只有一个会为True
                    infos['win1'] = True
                if any(self.red_team.alives) is False:
                    infos['lose1'] = True
            elif tru: #如果是因为截断结束的
                hp_red = sum([t.hp for t in self.red_tanks])
                hp_blue = sum([t.hp for t in self.blue_tanks])
                if hp_red > hp_blue:
                    infos['win2'] = True
                if hp_red <= hp_blue:
                    infos['lose2'] = True
            if infos['win1'] or infos['win2']:
                infos['win'] = True
            elif infos['lose1'] or infos['lose2']:
                infos['lose'] = True

        # 传出红蓝方的hp值
        for t in self.tanks + self.soldiers:
            infos[t.uid] = t.hp
        # 传出红蓝方的坐标
        for t in self.tanks + self.soldiers:
            infos[t.uid + '_xy'] = [int(t.x), int(t.y)]
        
        # 传出红蓝方坦克的燃油值
        for t in self.tanks:
            infos[t.uid + '_fuel'] = t.fuel

        return infos
    
    def step(
        self, actions: dict[str, Action]
    ) -> tuple[Observation, Reward, bool, bool, Info]:
        """环境迭代一个时间步，并返回五元组（观测，奖励，终止，截断，额外信息）。

        :param actions: 一个字典，用来表征坦克的动作，键是坦克的uid，值是动作，由动作类型和动作值两部分组成，支持仅传入部分坦克的动作
        在 step 方法中，所有智能体的动作是并发执行的，但环境更新和状态观测是作为一个整体来处理的
        """
        self.step_ += 1 #得在最前面 get_info()中用到
        num = np.random.randint(2) # 双方随机先后动作
        if num == 0:
            self.blue_policy(self.policy,self.noise_a,self.noise_b) #蓝方先动作 
            self.assign_actions(actions)
        elif num == 1:
            self.assign_actions(actions)
            self.blue_policy(self.policy,self.noise_a,self.noise_b)
        
        if self.render_mode == True:
            self.render() #

        r = self.get_reward() #消耗的弹药统一在这清除
        obs = self.get_obs()
        info = self.get_info()

        return obs, r, self.terminated, self.truncated, info

    def reset(self,policy_num=1,noise_a = 0.1*2*math.pi, noise_b = 0.12*2*math.pi) -> tuple[Observation, Info]:
        """重置环境，各个坦克返回原位置，且生命值完整，弹药量完整。"""
        for k in Tank.id_counter:
            Tank.id_counter[k] = 0
        for k in Soldier.id_counter:
            Soldier.id_counter[k] = 0
        self.red_team = Team("Red")
        self.blue_team = Team("Blue")
        self.step_ = 0  # 记录当前回合数
        
        self.policy = policy_num #蓝方策略
        self.noise_a = noise_a
        self.noise_b = noise_b

        info = {}
        if self.render_mode == True: #显示一个回合数0
            self.render()
        return self.get_obs(), info
    
    '''
    类似gym的接口
    state_dim = env.observation_space("Red-0").shape[0]
    action_dim = env.action_space("Red-0").shape[0]
    action = {agent: env.action_space(agent).sample() for agent in env.agents}
    '''
    def action_space(self, agent_id) -> spaces.Space:
        action = self.sample()
        return spaces.Box(-1, 1, shape=(len(action[agent_id]),), dtype=np.float32)

    def observation_space(self, agent_id) -> spaces.Space:
        obs = self.get_obs()
        return spaces.Box(0, 1, shape=(len(obs[agent_id]),), dtype=np.float32)
    
    def Normalization(self,obs):

        '''
        例： 将此值 归一化 {'Red-0': array([ 25.,   0.,  10.,   0.,   9., 200.,  25.,  50.,  15.,  50.,  35.,
        50.], dtype=float32), 'Red-1': array([ 15.,   0.,  10.,   0.,   9., 200.,  25.,  50.,  15.,  50.,  35.,
        50.], dtype=float32), 'Red-2': array([ 35.,   0.,  10.,   0.,   9., 200.,  25.,  50.,  15.,  50.,  35.,
        50.], dtype=float32)}
        '''

        for k in obs.keys():
            
            if 'soldier' in k:
                obs[k][0] = (obs[k][0] - config.observation_space["x"][0]) / (config.observation_space["x"][1] - config.observation_space["x"][0])
                obs[k][1] = (obs[k][1] - config.observation_space["y"][0]) / (config.observation_space["y"][1] - config.observation_space["y"][0])
                obs[k][2] = (obs[k][2] - config.observation_space["hp_s"][0]) / (config.observation_space["hp_s"][1] - config.observation_space["hp_s"][0])
                obs[k][3] = (obs[k][3] - config.observation_space["speed_s"][0]) / (config.observation_space["speed_s"][1] - config.observation_space["speed_s"][0])
                obs[k][4] = (obs[k][4] - config.observation_space["bullet_s"][0]) / (config.observation_space["bullet_s"][1] - config.observation_space["bullet_s"][0])
                # 动态调整 obs 后面都是位置
                n = len(obs[k])
                for i in range(5, n, 2): # 5 7 9 11
                    obs[k][i] = (obs[k][i] - config.observation_space["x"][0]) / (config.observation_space["x"][1] - config.observation_space["x"][0])
                    obs[k][i+1] = (obs[k][i+1] - config.observation_space["y"][0]) / (config.observation_space["y"][1] - config.observation_space["y"][0])
            
            elif 'tank' in k:
                obs[k][0] = (obs[k][0] - config.observation_space["x"][0]) / (config.observation_space["x"][1] - config.observation_space["x"][0])
                obs[k][1] = (obs[k][1] - config.observation_space["y"][0]) / (config.observation_space["y"][1] - config.observation_space["y"][0])
                obs[k][2] = (obs[k][2] - config.observation_space["hp_t"][0]) / (config.observation_space["hp_t"][1] - config.observation_space["hp_t"][0])
                obs[k][3] = (obs[k][3] - config.observation_space["speed_t"][0]) / (config.observation_space["speed_t"][1] - config.observation_space["speed_t"][0])
                obs[k][4] = (obs[k][4] - config.observation_space["bullet_t"][0]) / (config.observation_space["bullet_t"][1] - config.observation_space["bullet_t"][0])
                obs[k][5] = (obs[k][5] - config.observation_space["fuel"][0]) / (config.observation_space["fuel"][1] - config.observation_space["fuel"][0])

                # 动态调整 obs
                n = len(obs[k])
                for i in range(6, n, 2):
                    obs[k][i] = (obs[k][i] - config.observation_space["x"][0]) / (config.observation_space["x"][1] - config.observation_space["x"][0])
                    obs[k][i+1] = (obs[k][i+1] - config.observation_space["y"][0]) / (config.observation_space["y"][1] - config.observation_space["y"][0])

        return obs
    
    def init_render(self):
        #plt.ioff()  # 关闭交互模式
        plt.show()
        plt.ion() #打开交互模式
        self.fig,self.ax = plt.subplots(figsize = (7.5,5)) #ax是子图 #宽度为8英寸，高度为5英寸
        self.fig.canvas.manager.set_window_title(f'{config.team.red_n}vs{config.team.blue_n}坦克对战')
        self.ax.set_xlim(0, config.map.width)
        self.ax.set_ylim(0, config.map.height)
        self.ax.set_aspect('equal') #设置坐标轴比例 'equal'表示x轴y轴比例相等
        # 设置标题
        #self.ax.set_title('坦克对战')
        self.ax.set_xticks(range(0, config.map.width + 1, 10)) #设置坐标轴刻度
        self.ax.set_yticks(range(0, config.map.height + 1, 10)) #设置坐标轴刻度
        self.ax.grid(True) #显示网格
        
        self.jia = False
        self.r_color = 'r' if self.jia == False else 'b'
        self.b_color = 'b' if self.jia == False else 'r'
        self.r_label = '红方坦克' if self.jia == False else '蓝方坦克'
        self.b_label = '蓝方坦克' if self.jia == False else '红方坦克'
        self.r_bullet = '红方炮弹' if self.jia == False else '蓝方炮弹'
        self.b_bullet = '蓝方炮弹' if self.jia == False else '红方炮弹'
        self.r_ ='红' if self.jia == False else '蓝'
        self.b_ ='蓝' if self.jia == False else '红'

        ## 显示图例 #maker表示图例的样式,markerfacecolor表示图例的颜色,markersize表示图例的大小,label表示图例的标签,color表示图例的颜色
        self.ax.legend(handles=[plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=self.r_color, markersize=10, label=self.r_label),
                                plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=self.b_color, markersize=10, label=self.b_label),
                                plt.Line2D([0], [0], marker='.', color='w', markerfacecolor=self.r_color, markersize=10, label=self.r_bullet),
                                plt.Line2D([0], [0], marker='.', color='w', markerfacecolor=self.b_color, markersize=10, label=self.b_bullet )],bbox_to_anchor=(1.4, 1))      
        
    def render(self):
        
        
        #右上角显示回合数 
        self.ax.text(18, 51, f"回合数: {self.step_}/{config.max_steps}", fontsize=12)
        # 显示其他值
        #self.ax.text(50, 48 , f"    血量 炮弹 燃油" )
        
        stars =[]
        for t in self.red_tanks:
            n = int(t.uid[-1])
            _alpha = t.hp / config.tank.hp if t.hp / config.tank.hp > 0.5 else 0.5
            # 显示其他值
           # self.ax.text(50, 45-n*2 , f"{self.r_}{n} {t.hp:.1f} {len(t.bullets)}   {t.fuel}",)
            self.ax.text(t.x, t.y+0.5, f"{n}", fontsize=6, color='y', ha='center', va='center')
            if t.alive:
                circle = Circle((t.x, t.y), 1, color=self.r_color, fill=True,alpha=_alpha)
                self.ax.add_patch(circle)
                # 显示血量
                self.ax.text(t.x, t.y, f"{(t.hp):.0f}", fontsize=8, color='w', ha='center', va='center')

            else:
                circle = Circle((t.x, t.y), 1, color=self.r_color, fill=False)
                self.ax.add_patch(circle)
                self.ax.text(t.x, t.y, f"{(t.hp):.0f}", fontsize=8, color=self.r_color, ha='center', va='center')
                


            for b in t.consumed_bullets:
                b = Circle((b.x, b.y), 5, color=self.r_color, fill=False,linestyle='--')
                self.ax.add_patch(b)
                star = self.ax.plot(b.center[0], b.center[1], marker='.', color=self.r_color)
                stars.append(star)

        
        for t in self.blue_tanks:
            n = int(t.uid[-1])
            _alpha = t.hp / config.tank.hp if t.hp / config.tank.hp > 0.5 else 0.5
            #self.ax.text(50, 30-n*2 , f"{self.b_}{n} {t.hp:.1f} {len(t.bullets)}  {t.fuel}",)
            self.ax.text(t.x, t.y+0.5, f"{n}", fontsize=6, color='y', ha='center', va='center')
            if t.alive :
                circle = Circle((t.x, t.y), 1, color=self.b_color, fill=True,alpha=_alpha)
                self.ax.add_patch(circle)
                self.ax.text(t.x, t.y, f"{(t.hp):.0f}", fontsize=8, color='w', ha='center', va='center')
            else:
                circle = Circle((t.x, t.y), 1, color=self.b_color, fill=False)
                self.ax.add_patch(circle)
                self.ax.text(t.x, t.y, f"{(t.hp):.0f}", fontsize=8, color=self.b_color, ha='center', va='center')
            
            for b in t.consumed_bullets:
                b = Circle((b.x, b.y), 5, color=self.b_color, fill=False,linestyle='--')
                self.ax.add_patch(b)
                star = self.ax.plot(b.center[0], b.center[1], marker='.', color=self.b_color)
                stars.append(star)
        # 画表
        for  t in self.tanks:
            self.ax.table(
                            cellText=[
                    (f"{t.hp:.2f}", len(t.bullets), int(t.fuel)) for t in self.tanks
                ],
                rowLabels=[self.r_+t.uid[-1] for t in self.red_tanks]+[self.b_+t.uid[-1] for t in self.blue_tanks],
                colLabels=["生命值", "弹药量", "燃油量"],
                colWidths=[0.10, 0.10, 0.10],
                rowLoc="center",
                cellLoc="center",
                bbox=[1.1, 0, 0.30, 0.60],
            )
        # 士兵
        for t in self.red_team.soldiers:
            n = int(t.uid[-1])
            _alpha = t.hp / config.soldier.hp if t.hp / config.soldier.hp > 0.5 else 0.5
            # 显示其他值
            #self.ax.text(50, 45-n*2 , f"{self.r_}{n} {t.hp:.1f} {len(t.bullets)}   {t.fuel}",)
            self.ax.text(t.x, t.y+0.5, f"{n}", fontsize=6, color='y', ha='center', va='center')
            if t.alive:
                circle = Circle((t.x, t.y), 0.5, color=self.r_color,fill = True,alpha=_alpha)
                self.ax.add_patch(circle)
                # 显示血量
                #self.ax.text(t.x, t.y, f"{(t.hp):.0f}", fontsize=8, color='w', ha='center', va='center')

            else:
                circle = Circle((t.x, t.y), 0.5, color=self.r_color, fill = False)
                self.ax.add_patch(circle)
                #self.ax.text(t.x, t.y, f"{(t.hp):.0f}", fontsize=8, color=self.r_color, ha='center', va='center')
                

            for b in t.consumed_bullets:
                self.ax.plot([b.x, t.x], [b.y, t.y], ls="--", color=self.r_color)

        for t in self.blue_team.soldiers:
            n = int(t.uid[-1])
            _alpha = t.hp / config.soldier.hp if t.hp / config.soldier.hp > 0.5 else 0.5
            #self.ax.text(50, 30-n*2 , f"{self.b_}{n} {t.hp:.1f} {len(t.bullets)}  {t.fuel}",)
            self.ax.text(t.x, t.y+0.5, f"{n}", fontsize=6, color='y', ha='center', va='center')
            if t.alive :
                circle = Circle((t.x, t.y), 0.5, color=self.b_color, fill = True,alpha=_alpha)
                self.ax.add_patch(circle)
                #self.ax.text(t.x, t.y, f"{(t.hp):.0f}", fontsize=8, color='w', ha='center', va='center')
            else:
                circle = Circle((t.x, t.y), 0.5, color=self.b_color, fill = False)
                self.ax.add_patch(circle)
                #self.ax.text(t.x, t.y, f"{(t.hp):.0f}", fontsize=8, color=self.b_color, ha='center', va='center')
            
            for b in t.consumed_bullets:
                self.ax.plot([b.x, t.x], [b.y, t.y], ls="--", color=self.b_color)
        
        self.fig.canvas.flush_events() #flush_events()刷新图形
        self.fig.canvas.draw_idle()
        time.sleep(1) #延时1s

        # # 保存图片
        # # 创建一个文件夹
        # if not os.path.exists('test'):
        #      os.makedirs('test')
        # plt.savefig(f'test/test{self.step_}.jpg')
        
        # 清除画面
        text = [t for t in self.ax.texts]
        for text in text:
            text.remove()
        #text.remove()
        if len(stars) > 0:
            for star_list in stars:
                for star in star_list:  # star_list 是一个列表，所以我们遍历它
                    star.remove()

        # if self.render_mode == True:
        #     for t in self.tanks:
        #         t.consumed_bullets.clear()
        for _t in self.ax.tables:
            _t.remove()

        for patch in self.ax.patches:
            patch.remove()
        for line in self.ax.lines:
            line.remove()

        #plt.pause(0.01) #
        #显示图像

        #plt.draw() #draw()重新绘制图形
    

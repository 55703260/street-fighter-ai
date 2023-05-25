import math
import time
import collections

import gym
import numpy as np

# Custom environment wrapper
class StreetFighterCustomWrapper(gym.Wrapper):
    def __init__(self, env, reset_round=True, rendering=False):
        super(StreetFighterCustomWrapper, self).__init__(env)
        self.env = env

        # Use a deque to store the last 9 frames
        self.num_frames = 9
        self.frame_stack = collections.deque(maxlen=self.num_frames)

        self.num_step_frames = 6

        self.reward_coeff = 3.0

        self.total_timesteps = 0

        self.full_hp = 176
        self.prev_player_health = self.full_hp
        self.prev_oppont_health = self.full_hp

        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(100, 128, 3), dtype=np.uint8)
        
        self.reset_round = reset_round
        self.rendering = rendering
    
    # 这个函数 _stack_observation 的目的是将多帧的游戏画面叠加起来形成一个更复杂的观察值。首先，我们看 self.frame_stack[i * 3 + 2][:, :, i] 这部分。
    # 这里，self.frame_stack 是一个包含了之前几帧游戏画面的队列。i * 3 + 2 是索引，由于 i 在 range(3) 中，所以 i * 3 + 2 的取值会是 2, 5, 8。
    # 因此，这段代码会取出 self.frame_stack 中索引为 2, 5, 8 的三帧画面。每一帧画面是一个三维的张量，其中第三个维度是颜色通道（RGB）。
    # 接下来，[:, :, i] 这部分，会从每一帧画面中取出第 i 个颜色通道的值。由于 i 在 range(3) 中，所以 i 的取值会是 0, 1, 2，分别对应 RGB 中的 R、G、B 三个颜色通道。
    # 最后，np.stack([self.frame_stack[i * 3 + 2][:, :, i] for i in range(3)], axis=-1) 这部分，会将取出的三个颜色通道的值沿着第三个维度（颜色通道维度）堆叠起来。
    # 换句话说，这段代码会从索引为 2, 5, 8 的三帧画面中，分别取出 R、G、B 三个颜色通道的值，然后将这三个颜色通道的值组合成一个新的画面。
    # 这样，每次调用 _stack_observation 函数，都会返回一个包含了三帧画面信息的新画面。这种方法有助于模型学习游戏中的动态信息，比如角色的移动方向、速度等等。
    def _stack_observation(self):
        return np.stack([self.frame_stack[i * 3 + 2][:, :, i] for i in range(3)], axis=-1)

    def reset(self):
        observation = self.env.reset()
        
        self.prev_player_health = self.full_hp
        self.prev_oppont_health = self.full_hp

        self.total_timesteps = 0
        
        # Clear the frame stack and add the first observation [num_frames] times
        self.frame_stack.clear()
        for _ in range(self.num_frames):
            self.frame_stack.append(observation[::2, ::2, :]) #沿着第一维度和第二维度每隔一个元素取一个元素，而在第三维度取所有的元素,相当于对图像进行了降采样，使其宽度和高度都缩小了一半

        return np.stack([self.frame_stack[i * 3 + 2][:, :, i] for i in range(3)], axis=-1)

    def step(self, action):
        custom_done = False

        obs, _reward, _done, info = self.env.step(action)
        self.frame_stack.append(obs[::2, ::2, :])

        # Render the game if rendering flag is set to True.
        if self.rendering:
            self.env.render()
            time.sleep(0.01)  #游戏减速

        for _ in range(self.num_step_frames - 1):
            
            # Keep the button pressed for (num_step_frames - 1) frames.
            obs, _reward, _done, info = self.env.step(action)
            self.frame_stack.append(obs[::2, ::2, :])
            if self.rendering:
                self.env.render()
                time.sleep(0.01) #游戏减速

        curr_player_health = info['agent_hp']
        curr_oppont_health = info['enemy_hp']
        
        self.total_timesteps += self.num_step_frames
        
        # Game is over and player loses.
        if curr_player_health < 0:
            custom_reward = -math.pow(self.full_hp, (curr_oppont_health + 1) / (self.full_hp + 1))    # Use the remaining health points of opponent as penalty. 
                                                   # If the opponent also has negative health points, it's a even game and the reward is +1.
            custom_done = True

        # Game is over and player wins.
        elif curr_oppont_health < 0:
            # custom_reward = curr_player_health * self.reward_coeff # Use the remaining health points of player as reward.
                                                                   # Multiply by reward_coeff to make the reward larger than the penalty to avoid cowardice of agent.

            # custom_reward = math.pow(self.full_hp, (5940 - self.total_timesteps) / 5940) * self.reward_coeff # Use the remaining time steps as reward.
            custom_reward = math.pow(self.full_hp, (curr_player_health + 1) / (self.full_hp + 1)) * self.reward_coeff
            custom_done = True

        # While the fighting is still going on
        else:
            custom_reward = self.reward_coeff * (self.prev_oppont_health - curr_oppont_health) - (self.prev_player_health - curr_player_health)
            self.prev_player_health = curr_player_health
            self.prev_oppont_health = curr_oppont_health
            custom_done = False

        # When reset_round flag is set to False (never reset), the session should always keep going.
        if not self.reset_round:
            custom_done = False
             
        # Max reward is 6 * full_hp = 1054 (damage * 3 + winning_reward * 3) norm_coefficient = 0.001
        return self._stack_observation(), 0.001 * custom_reward, custom_done, info # reward normalization
    
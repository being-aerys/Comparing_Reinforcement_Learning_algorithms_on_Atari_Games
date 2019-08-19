'''An attempt to solve playing Pong sing DQN following the book my Lapan. The book follows the original papers by the authors of DQN
and its variants'''

import time, gym, collections
import numpy as np



'''The following wrapper presses the FIRE button in environments that require them for the game to start. Also, it checks for
several corner cases that are present in some games.'''
class Fire_On_Env_Reset(gym.Wrapper):
    def __init__(self, env = None):
        super(Fire_On_Env_Reset, self).__init__(env)
        assert env.unwrapped.get_action_meanings()[1] == "FIRE" #asserting that this env has a FIRE action.
        assert len(env.unwrapped.get_action_meanings()) >= 3 #asserting that there are at  least 3 or more actions, not sure why we
                                                             #need this though

        #env.unwrapped() returns the base unwrapped instance of this env.
        #get_action_meanings() returns the meanings of different actions that can be taken in this env as shown in the
        #footer.

    def step(self, action):
        return self.env.step(action)

    def reset(self):
        self.env.reset()

        #Not sure why we are doing th following.
        obs, _, done, _ = self.env.step(1) #may be 1 is left for PONG
        if done:
            self.env.reset()

        obs, _, done, _ = self.env.step(2) #may be 2 is right for PONG
        if done:
            self.env.reset()

        return obs

'''The following wrapper returns only every the n consecutive timesteps of the gameplay. It takes a maxpool of the recent 2 time steps 
frames and returns it as the obs.. Why does this work though?'''

class Skip_Frame_Max_Env(gym.Wrapper):

    def __init__(self, env = None, skip = 4):
        super(Skip_Frame_Max_Env, self).__init__(env)
        self.obs_buffer = collections.deque(maxlen= 2)
        # Once a bounded length deque is full, when new items are added,
        # a corresponding number of items are discarded from the opposite end.
        self.skip_at_these_many_steps = skip

    def step(self, action):
        total_reward = 0.0
        done = None

        #apply same action to all n frames assuming that these frames are somewhat identical
        for _ in range(self.skip_at_these_many_steps):
            obs, reward, done, info = self.env.step(action)
            self.obs_buffer.append(obs)
            total_reward += reward
            if done:
                break
        max_frame = np.max(np.stack(self.obs_buffer), axis = 0)# taking the max from the last two time steps
        return max_frame, total_reward, done, info

    def reset(self):
        self.obs_buffer.clear()
        obs = self.env.reset()
        self.obs_buffer.append(obs)
        return obs


'''The following wrapper crops the emulator console output to 84 * 84 to remove irrelevant screen information. Also, it converts
RGB to grayscale since color does not matter for Atari games and grayscale is enough to distinguish the relevant information.'''


'''
Action meanings for a dummy environment look like this in gym.
ACTION_MEANING = {
    0: "NOOP",
    1: "FIRE",
    2: "UP",
    3: "RIGHT",
    4: "LEFT",
    5: "DOWN",
    6: "UPRIGHT",
    7: "UPLEFT",
    8: "DOWNRIGHT",
    9: "DOWNLEFT",
    10: "UPFIRE",
    11: "RIGHTFIRE",
    12: "LEFTFIRE",
    13: "DOWNFIRE",
    14: "UPRIGHTFIRE",
    15: "UPLEFTFIRE",
    16: "DOWNRIGHTFIRE",
    17: "DOWNLEFTFIRE",


'''
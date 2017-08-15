#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Sarsa is a online updating method for Reinforcement learning.

Unlike Q learning which is a offline updating method, Sarsa is updating while in the current trajectory.

You will see the sarsa is more coward when punishment is close because it cares about all behaviours,
while q learning is more brave because it only cares about maximum behaviour.
"""

from maze_env import Maze
from RL_brain import DeepQNetwork
import numpy as np


def run_maze():
    step = 0
    index_set = []
    for episode in range(500):
        # initial observation
        observation, index, action = env.reset()
        index_set.append(index)
        # print(episode)
        while True:

            # RL choose action based on observation
            action_ = RL.choose_action(observation, action)

            # RL take action and get next observation and reward
            observation_, reward, index_, action_, done = env.step(action_, index)

            if observation_ is not None and index_ is not None:
                RL.store_transition(observation, action_, reward, index, index_, observation_)

            if (step > 200) and (step % 5 == 0):
                RL.learn()

            # swap observation
            if observation_ is not None and index_ is not None:
                observation = observation_
                index = index_
                action = action_

            # break while loop when end of this episode
            if done:
                break
            step += 1

    # end of game
    print('game over step is {}'.format(step))
    print('set length is {}').format(len(set(index_set)))


def load_model():
    # x_data = np.asarray(env.batch_xs)
    i = 0
    for index in range(env.batch_xs.shape[0]):
        action = str(int(env.batch_xs.loc[index]['keyword'])) + '-' + str(
            int(env.batch_xs.loc[index]['delivery_times']))
        action_ = RL.predict_action(env.batch_xs.loc[index])
        action_ = env.action_space[action_]
        if action == action_:
            i += 1
        print action_
    print i/env.batch_xs.shape[0]



if __name__ == "__main__":
    # maze game
    env = Maze()
    RL = DeepQNetwork(env.n_actions, env.n_features,
                      learning_rate=0.01,
                      reward_decay=0.9,
                      e_greedy=0.9,
                      replace_target_iter=200,
                      memory_size=2000,
                      output_graph=True
                      )
    # run_maze()
    # RL.save_model()
    # RL.plot_cost()
    RL.load_model()
    load_model()

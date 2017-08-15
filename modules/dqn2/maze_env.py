#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import pandas as pd
import numpy as np
import time
import tensorflow as tf
from sklearn.externals.joblib import load, dump

project_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
sys.path.insert(0, project_path)

storage_path = project_path + '/storage/'

np.random.seed(2)  # reproducible

UNIT = 40  # pixels
MAZE_H = 4  # grid height
MAZE_W = 4  # grid width


class Maze(object):
    def __init__(self):

        self.batch_xs = pd.read_csv(storage_path + 'nctr/train_x_dul.csv')
        self.batch_ys = pd.read_csv(storage_path + 'nctr/train_y_dul.csv')
        self.action_index = ['keyword', 'delivery_times']
        self.action_cache = []
        self.action = self.get_action()
        self.action_space = self.merge_action()
        self.n_actions = len(self.action_space)
        self.n_features = 23

    def reset(self):
        x_data = np.asarray(self.batch_xs)
        index = np.random.randint(0, x_data.shape[0])
        # 下一步的正确动作
        yx = self.batch_ys.loc[index]
        action_ = '-'.join([str(yx['keyword'].astype(int)), str(yx['delivery_times'].astype(int))])
        action_ = self.action_space.index(action_)
        return x_data[index], index, action_

    def get_action(self):
        action = {}
        action_index = ['age', 'education', 'keyword', 'region', 'delivery_times']
        dicLoad = load(storage_path + 'nctr/train.dat')
        for i in action_index:
            action[i] = [j for j in range(len(dicLoad[i]))]
        return action

    def merge_action(self, n=0, loca=None):
        for j in self.action[self.action_index[n]]:
            if n + 1 <= len(self.action_index) - 1:
                if loca == None:
                    self.merge_action(n + 1, j)
                else:
                    self.merge_action(n + 1, str(loca) + '-' + str(j))
            else:
                self.action_cache.append(str(loca) + '-' + str(j))

        # return '-'.join(state_index)
        return self.action_cache

    def step(self, action, index):

        action1 = self.action_space[action]
        action1 = action1.split('-')
        # ctr = self.batch_xs.loc[index].ix['ctr']
        next = self.batch_ys.loc[index]
        predict = next.ix['ctr']
        # reward function
        if str(int(next.ix[self.action_index[0]])) == action1[0] and str(int(next.ix[self.action_index[1]])) == action1[
            1]:
            s_ = next
            index_ = list(np.where(
                (self.batch_xs["ctr"] == next.ix['ctr']) & (self.batch_xs["bid_amount"] == next.ix['bid_amount']) & (
                    self.batch_xs["pv"] == next.ix['pv']) & (self.batch_xs["click"] == next.ix['click']) & (
                    self.batch_xs["cost"] == next.ix['cost']) & (
                    self.batch_xs["iAllMinute"] == next.ix['iAllMinute']))[0])
            if len(index_) > 0:
                index_ = index_[0]
                yx = self.batch_ys.loc[index_]
                action_ = '-'.join([str(yx['keyword'].astype(int)), str(yx['delivery_times'].astype(int))])
                action_ = self.action_space.index(action_)
                done = False
            else:
                index_ = None
                action_ = None
                done = True

            if predict < 0.03:
                reward = -1
            elif predict >= 0.03 and predict < 0.04:
                reward = 1
            elif predict >= 0.04 and predict < 0.05:
                reward = 2
            elif predict >= 0.05 and predict < 0.06:
                reward = 3
            elif predict >= 0.06 and predict < 0.07:
                reward = 4
            elif predict >= 0.07 and predict < 0.08:
                reward = 5
            else:
                reward = 6
        else:
            s_ = None
            reward = 0
            index_ = None
            action_ = None
            done = False

        # if index_ == self.batch_xs.shape[0]:
        #     done = True

        return s_, reward, index_, action_, done

    def step2(self, action, index):

        action1 = self.action_space[action]
        action1 = action1.split('-')
        # ctr = self.batch_xs.loc[index].ix['ctr']
        next = self.batch_ys.loc[index]
        predict = next.ix['ctr']
        # reward function
        if str(int(next.ix[self.action_index[0]])) == action1[0] and str(int(next.ix[self.action_index[1]])) == action1[
            1]:
            s_ = self.batch_ys.loc[index]
            index_ = list(np.where(
                (self.batch_xs["ctr"] == next.ix['ctr']) & (self.batch_xs["bid_amount"] == next.ix['bid_amount']) & (
                    self.batch_xs["pv"] == next.ix['pv']) & (self.batch_xs["click"] == next.ix['click']) & (
                    self.batch_xs["cost"] == next.ix['cost']) & (
                    self.batch_xs["iAllMinute"] == next.ix['iAllMinute']))[0])
            if len(index_) > 0:
                index_ = index_[0]
                yx = self.batch_ys.loc[index_]
                action_ = '-'.join([str(yx['keyword'].astype(int)), str(yx['delivery_times'].astype(int))])
                action_ = self.action_space.index(action_)
                done = False
            else:
                index_ = None
                action_ = None
                done = True

            if predict < 0.03:
                reward = -1
            elif predict >= 0.03 and predict < 0.04:
                reward = 1
            elif predict >= 0.04 and predict < 0.05:
                reward = 2
            elif predict >= 0.05 and predict < 0.06:
                reward = 3
            elif predict >= 0.06 and predict < 0.07:
                reward = 4
            elif predict >= 0.07 and predict < 0.08:
                reward = 5
            else:
                reward = 6
        else:
            s_ = None
            reward = 0
            index_ = None
            action_ = None
            done = False

        # if index_ == self.batch_xs.shape[0]:
        #     done = True

        return s_, reward, index_, action_, done

# as learned on: https://www.learndatasci.com/tutorials/reinforcement-q-learning-scratch-python-openai-gym/
# implement Q-table

import gym
import numpy as np
import random
from time import sleep
import matplotlib.pyplot as plt

class tutorial():
    def __init__(self, epochs=100_000, steps=1_000):
        self.env = gym.make('Taxi-v3').env
        self.epochs = epochs
        self.qtable = np.zeros([self.env.observation_space.n, self.env.action_space.n])

        # hyperparameters
        self.alpha = 0.1
        self.gamma = 0.6
        self.epsilon = 0.1

        # metrics plotting
        self.all_timesteps, self.all_penalties = [], []

    def run(self):
        self.train()
        self.evaluate()
        self.play()

    def train(self):
        for epoch in range(self.epochs):
            state = self.env.reset()
            steps, reward, penalties = 0, 0, 0
            done = False
            while not done:
                if random.uniform(0,1) < self.epsilon:
                    action = self.env.action_space.sample()
                else:
                    action = np.argmax(self.qtable[state])

                next_state, reward, done, info = self.env.step(action)

                old_value = self.qtable[state, action]
                next_max = np.max(self.qtable[next_state])

                new_value = (1 - self.alpha) * old_value + self.alpha * (reward + self.gamma * next_max)
                self.qtable[state, action] = new_value

                if reward == -10: penalties += 1

                state = next_state
                steps += 1
            self.all_timesteps.append(steps)
            self.all_penalties.append(penalties)
            if epoch%(self.epochs/10) == 0:
                print(f'Epoch: {epoch}')
        print('Training Completed')

    def evaluate(self):
        plt.figure(0)
        plt.title('Timesteps per Epoch')
        plt.plot(range(self.epochs), self.all_timesteps)
        plt.show(block=0)
        plt.figure(1)
        plt.title('Penalties per Epoch')
        plt.plot(range(self.epochs), self.all_penalties)
        plt.show(block=1)

        print(f'Total Epochs: {self.epochs:,}')
        print(f'average timesteps per epoch: {np.average(self.all_timesteps)}')
        print(f'average penalties per epoch: {np.average(self.all_penalties)}')

    def play(self):
        state = self.env.reset()
        steps, reward, penalties = 0, 0, 0
        done = False
        while not done:
            action = np.argmax(self.qtable[state])
            next_state, reward, done, info = self.env.step(action)
            if reward == -10: penalties += 1
            state = next_state
            steps += 1
            self.env.render()
            sleep(0.1)

    def explore_state(self, state=[3,1,2,0]):
        print(f'state: {state}')
        self.env.s = self.env.encode(*state)
        self.env.render()

    def explore_reward_table(self, state=328):
        print(f'state: {state}')
        print(self.env.P(state))


if __name__ ==  '__main__':
    t = tutorial()
    t.run()
# Getting started: http://gym.openai.com/docs/
# what this does:
# start environment and get agent to do a random action
# agent DOES NOT learn in this example

import gym

class tutorial():
    def __init__(self, iterations=200, steps=10_000):
        self.env = gym.make('CartPole-v0')
        self.iterations = iterations
        self.steps = steps

    def run(self):
        for i_episodes in range(self.iterations):
            observation = self.env.reset()
            for t in range(self.steps):
                self.env.render()
                print(observation)
                action = self.env.action_space.sample()
                observation, reward, done, info = self.env.step(action)
                if done:
                    print(f'Episode {i_episodes} finished after {t+1} timesteps')
                    break
        self.env.close()

if __name__ == '__main__':
    g = tutorial()
    g.run()
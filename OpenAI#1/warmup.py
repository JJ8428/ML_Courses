import sys
import gym
import keras
import random
import math
import numpy as np
from collections import deque
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

# Set up the OpenAI
env = gym.make('CartPole-v0')

for i_episode in range(20):
    observation = env.reset()
    for t in range(1000):
        env.render()
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)

        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break

print(env.action_space)
print(env.observation_space)

# Check bounds of box(space)

print(env.observation_space.high)
print(env.observation_space.low)

n_episodes = 1000 # Steps from beginning to end (or fail)
n_win_ticks = 195 # Number of attempts to be made with 1000 steps in each trial
max_env_steps = None

gamma = 1.0 # Discout factor is future reward (Weight of reward)
epsilon = 1.0 # Exploration factor (Choose a uniformly random choice to see it if picks a better choice)
epsilon_min = 0.01
epsilon_decay = .995
alpha = .01 # Learning rate, determines how much new data overwrites old data
alpha_decay = .01

batch_size=64
monitor=False
quiet=False

# Environment Parameters
memory = deque(maxlen=100000)
env = gym.make('CartPole-v0')
if max_env_steps is not None: env.max_episode_steps = max_env_steps

# Creating the model
# Hidden layers
model = Sequential()
model.add(Dense(24, input_dim=4, activation='relu'))
model.add(Dense(48, activation='relu'))
model.add(Dense(2, activation='relu')) # 2 possible outputs, either left or right
model.compile(loss='mse', optimizer=Adam(lr=alpha, decay=alpha_decay))

# Define some necessary functions

def remember (state, action, reward, next_state, done):
    memory.append((state, action, reward, next_state, done))

def choose_action(state, epsilon):
    return env.action_space.sample() if (np.random.random() <= epsilon) else np.argmax(model.predict(state))

def get_epsilon(t):
    return max(epsilon_min, min(epsilon, 1.0 - math.log10((t+1)*epsilon_decay)))

def preprocess_state(state):
    return np.reshape(state, [1, 4])

def replay(batch_size, epsilon):
    x_batch, y_batch = [], []
    minibatch = random.sample(memory, min(len(memory), batch_size))

    for state, action, reward, next_state, done in minibatch:
        y_target = model.predict(state)
        # Reward is issued if choice(random) is correct
        y_target[0][action] = reward if done else reward + gamma * np.max(model.predict(next_state)[0])
        x_batch.append(state[0])
        y_batch.append(y_target[0])

    model.fit(np.array(x_batch), np.array(y_batch), batch_size=len(x_batch), verbose=0)

    # Reduce exploration over time
    if epsilon > epsilon_min:
        epsilon *= epsilon_decay     

def run():
    scores = deque(maxlen=100)

    for e in range(n_episodes):
        state = preprocess_state(env.reset())
        done = False
        i = 0
        while not done:
            action = choose_action(state, get_epsilon(e))
            next_state, reward, done, _ = env.step(action)
            env.render()
            next_state = preprocess_state(next_state)
            remember(state, action, reward, next_state, done)
            state = next_state
            i += 1

        scores.append(i)
        mean_score = np.mean(scores)
        if mean_score >= n_win_ticks and e >= 100:
            if not quiet: print('Ran {} episodes. Solved after {} trials'.format(e, e - 100))
            return e - 100
        if e % 20 == 0 and not quiet:
            print('[Episode {}] - Mean survival time over last 100 episodes was {} ticks.'.format(e, mean_score))
   
        replay(batch_size, get_epsilon(e))

    if not quiet: print('Did not solve after {} episodes'.format(e))
    return e

run()

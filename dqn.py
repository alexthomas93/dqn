# TODO: Suppress NumPy warnings
import os
from collections import deque
from random import sample

import gym
import numpy as np
from keras import Sequential
from keras.layers import Dense
from keras.models import load_model
from keras.optimizers import Adam
from moviepy.editor import ImageSequenceClip

import defaults

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


class Agent:
    def __init__(self, env=defaults.env, learning_rate=defaults.learning_rate, batch_size=defaults.batch_size,
                 decay_delay=defaults.decay_delay, discount_factor=defaults.discount_factor, episodes=defaults.episodes,
                 epsilon=defaults.epsilon, epsilon_decay=defaults.epsilon_decay, epsilon_min=defaults.epsilon_min,
                 memory_len=defaults.memory_len, update_rate=defaults.update_rate):
        self.env = gym.make(env)
        self.state_size = self.env.observation_space.shape[0]
        self.num_actions = self.env.action_space.n
        self.learning_rate = learning_rate
        self.model = self.build_model()
        self.target_model = self.build_model()
        self.batch_size = batch_size
        self.decay_delay = decay_delay
        self.discount_factor = discount_factor
        self.episodes = episodes
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.memory = deque(maxlen=memory_len)
        self.update_rate = update_rate

    def build_model(self):
        model = Sequential()
        model.add(Dense(24, activation="relu", input_dim=self.state_size))
        model.add(Dense(24, activation="relu"))
        model.add(Dense(self.num_actions, activation="linear"))
        model.compile(Adam(self.learning_rate), "mean_squared_error")
        return model

    def act(self, state, epsilon_greedy=True):
        if np.random.rand() <= self.epsilon and epsilon_greedy:
            return self.env.action_space.sample()
        else:
            return np.argmax(self.model.predict(np.array([state])))

    def replay(self):
        if len(self.memory) > self.batch_size:
            minibatch = sample(self.memory, self.batch_size)
            for state, action, reward, next_state, done in minibatch:
                if done:
                    target = reward
                else:
                    target = reward + self.discount_factor * np.amax(self.target_model.predict(np.array([next_state])))
                x = np.array([state])
                y = np.array(self.model.predict(np.array([state])))
                y[0][action] = target
                self.model.fit(x, y, verbose=False)

    def decay_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def train_model(self, model=defaults.model):
        for episode in range(self.episodes):
            state = self.env.reset()
            done = False
            score = 0
            steps = 0
            while not done:
                action = self.act(state)
                next_state, reward, done, _ = self.env.step(action)
                score += reward
                steps += 1
                self.memory.append((state, action, reward, next_state, done))
                state = next_state
                self.replay()
                if steps % self.update_rate == 0:
                    self.update_target_model()
                if episode > self.decay_delay:
                    self.decay_epsilon()  # TODO: Should this be inside or outside the inner while loop?
            print("Episode: {}/{}\tScore: {}\tEpsilon: {}".format(episode, self.episodes, score, self.epsilon))
        self.model.save("models/{}.h5".format(model))

    def load_model(self, model=defaults.model):
        self.model = load_model("models/{}.h5".format(model))

    def run(self):
        state = self.env.reset()
        score = 0
        for _ in range(200):
            action = self.act(state, False)
            next_state, reward, done, _ = self.env.step(action)
            score += reward
            if done:
                break
            else:
                state = next_state
        print("Score: {}".format(score))

    def generate_gif(self, filename=defaults.filename):
        state = self.env.reset()
        frames = []
        for _ in range(200):
            action = self.act(state, False)
            frame = self.env.render("rgb_array")
            next_state, reward, done, _ = self.env.step(action)
            frames.append(frame)
            if done:
                break
            else:
                state = next_state
        gif = ImageSequenceClip(frames, fps=30)
        gif.write_gif("{}.gif".format(filename), program="imageio")

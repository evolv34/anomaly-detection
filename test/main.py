from collections import deque

import gym
import numpy as np
from tensorflow import keras
import tensorflow as tf

# temp history
history = deque(maxlen=2000)

# env specs
env = gym.make('CartPole-v0')
input_shape = [env.observation_space.shape[0]]
n_outputs = env.action_space.n

batch_size = 32
discount_factor = 0.95
optimizer = keras.optimizers.Adam(lr=0.0001)
loss_fn = keras.losses.mean_squared_error

tf.keras.backend.clear_session()
model = keras.models.Sequential([
    keras.layers.Dense(32, activation='elu', input_shape=input_shape),
    keras.layers.Dense(32, activation='elu'),
    keras.layers.Dense(n_outputs)
])

print(model.summary())


def epsilon_greedy_policy(state, epsilon_local=0):
    if np.random.rand() < epsilon_local:
        return np.random.randint(n_outputs)
    else:
        Q_values = model.predict(state[np.newaxis])
        return np.argmax(Q_values[0])


def play_one_step(env_local, state, epsilon_local):
    action = epsilon_greedy_policy(state, epsilon_local)
    next_state_local, reward_local, done_local, info_local = env_local.step(action)
    history.append((state, action, reward_local, next_state_local, done_local))
    return next_state_local, reward_local, done_local, info_local


def sample_experiences(batch_size_local):
    indices = np.random.randint(len(history), size=batch_size_local)
    batch = [history[index] for index in indices]
    states, actions, rewards, next_states, dones = [
        np.array([experience[field_index] for experience in batch]) for field_index in range(5)
    ]
    return states, actions, rewards, next_states, dones


def training_step(batch_size_local):
    experiences = sample_experiences(batch_size_local)
    states, actions, rewards, next_states, dones = experiences
    next_Q_values = model.predict(next_states)
    max_next_Q_values = np.max(next_Q_values, axis=1)
    target_Q_values = (rewards + discount_factor * max_next_Q_values)

    mask = tf.one_hot(actions, n_outputs)
    with tf.GradientTape() as tape:
        all_Q_values = model(states)
        Q_values = tf.reduce_sum(all_Q_values * mask, axis=1, keepdims=True)
        loss = tf.reduce_mean(loss_fn(target_Q_values, Q_values))

    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))


if __name__ == '__main__':
    print(input_shape)
    print(n_outputs)

    tf.keras.backend.clear_session()
    model = keras.models.Sequential([
        keras.layers.Dense(32, activation='elu', input_shape=input_shape),
        keras.layers.Dense(32, activation='elu'),
        keras.layers.Dense(n_outputs)
    ])

    for episode in range(600):
        print(f"executing episode = {episode}")
        obs = env.reset()

        for step in range(200):
            env.render()
            epsilon = max(1 - episode / 500, 0.01)
            obs, reward, done, info = play_one_step(env, obs, epsilon)
            if done:
                break

        if episode > 50:
            training_step(batch_size)

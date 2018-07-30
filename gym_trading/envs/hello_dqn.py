import gym

import gym_trading
import pandas as pd
import numpy as np
import trading_env as te
from collections import deque
import tensorflow as tf
import dqn
import random
from typing import List
import timeit

pd.set_option('display.width',500)

env = gym.make('trading-v0')
# env = gym.make('CartPole-v0')
 
#env.time_cost_bps = 0
env._max_episode_steps = None

env = gym.wrappers.Monitor(env, 'gym-results/', force=True)
INPUT_SHAPE = env.observation_space.shape
OUTPUT_SIZE = env.action_space.n

DISCOUNT_RATE = 0.99
REPLAY_MEMORY = 50000
MAX_EPISODES = 10
BATCH_SIZE = 64
TARGET_UPDATE_FREQUENCY = 5

def replay_train(mainDQN: dqn.DQN, targetDQN: dqn.DQN, train_batch: list) -> float:
    states = np.vstack([x[0] for x in train_batch])
    actions = np.array([x[1] for x in train_batch])
    rewards = np.array([x[2] for x in train_batch])
    next_states = np.vstack([x[3] for x in train_batch])
    done = np.array([x[4] for x in train_batch])

    X = states
    result = targetDQN.predict(next_states)
    Q_target = rewards + DISCOUNT_RATE * np.max(result, axis=1) * ~done

    y = mainDQN.predict(states)
    y[np.arange(int(len(X)/INPUT_SHAPE[0])), actions] = Q_target

    # Train our network using target and predicted Q values on each episode
    return mainDQN.update(X, y)


def get_copy_var_ops(*, dest_scope_name: str, src_scope_name: str) -> List[tf.Operation]:
    # Copy variables src_scope to dest_scope
    op_holder = []

    src_vars = tf.get_collection(
        tf.GraphKeys.TRAINABLE_VARIABLES, scope=src_scope_name)
    dest_vars = tf.get_collection(
        tf.GraphKeys.TRAINABLE_VARIABLES, scope=dest_scope_name)

    for src_var, dest_var in zip(src_vars, dest_vars):
        op_holder.append(dest_var.assign(src_var.value()))

    return op_holder


def bot_play(mainDQN: dqn.DQN, env: gym.Env) -> None:
    state = env.reset()
    reward_sum = 0

    while True:

        env.render()
        action = np.argmax(mainDQN.predict(state))
        state, reward, done, _ = env.step(action)
        reward_sum += reward

        if done:
            print("Total score: {}".format(reward_sum))
            break


def main():
    start = timeit.default_timer()
    # store the previous observations in replay memory
    replay_buffer = deque(maxlen=REPLAY_MEMORY)

    last_100_game_reward = deque(maxlen=100)

    with tf.Session() as sess:
        mainDQN = dqn.DQN(sess, INPUT_SHAPE, OUTPUT_SIZE, name="main")
        targetDQN = dqn.DQN(sess, INPUT_SHAPE, OUTPUT_SIZE, name="target")
        sess.run(tf.global_variables_initializer())

        # initial copy q_net -> target_net
        copy_ops = get_copy_var_ops(dest_scope_name="target",
                                    src_scope_name="main")
        sess.run(copy_ops)

        for episode in range(MAX_EPISODES):
            e = 1. / ((episode / 10) + 1)
            done = False
            step_count = 0
            state = env.reset()
            total_reward = 0
            last_info = None

            episode_start = timeit.default_timer()

            while not done:
                if np.random.rand() < e:
                    action = env.action_space.sample()
                else:
                    # Choose an action by greedily from the Q-network
                    action = np.argmax(mainDQN.predict(state))

                # Get new state and reward from environment
                next_state, reward, done, info = env.step(action)

                # if done:  # Penalty
                #     reward = -1

                # Save the experience to our buffer
                replay_buffer.append((state, action, reward, next_state, done))

                if len(replay_buffer) > BATCH_SIZE:
                    minibatch = random.sample(replay_buffer, BATCH_SIZE)
                    loss, train = replay_train(mainDQN, targetDQN, minibatch)
                    if step_count % 100 == 0:
                        print('\t{}/{}-Loss:{}, train:{}'.format(episode, step_count, loss, train))

                if step_count % TARGET_UPDATE_FREQUENCY == 0:
                    sess.run(copy_ops)

                state = next_state
                step_count += 1
                total_reward += reward
                last_info = info

            episode_stop = timeit.default_timer()
            print("Episode: {}  steps: {}, reward: {}, last_info:{}, durations:{}".format(episode, step_count, total_reward, last_info, episode_stop-episode_start))
            replay_buffer.clear()

            # CartPole-v0 Game Clear Checking Logic
            last_100_game_reward.append(step_count)

            if len(last_100_game_reward) == last_100_game_reward.maxlen:
                avg_reward = np.mean(last_100_game_reward)

                if avg_reward > 199:
                    print(f"Game Cleared in {episode} episodes with avg reward {avg_reward}")
                    break
        
        stop = timeit.default_timer()
        print('Estimate time {} for {} Episodes'.format(stop-start, MAX_EPISODES))

        bot_play(targetDQN, env)


if __name__ == "__main__":
    main()

# obs = []
# for _ in range(Episodes):
#     observation = env.reset()
#     done = False
#     count = 0
#     while not done:
#         action = env.action_space.sample() # random
#         observation, reward, done, info = env.step(action)
#         obs = obs + [observation]
#         #print observation,reward,done,info
#         count += 1
#         if done:
#             print(reward)
#             print(count)

# df = env.env.sim.to_df()

# print(df.head())
# print(df.tail())

# buyhold = lambda x,y : 2
# df = env.env.run_strat( buyhold )
# print('BuyHold\n{}'.format(df.tail()))

# randomtrader = lambda o,e: e.action_space.sample() # retail trader
# df = env.env.run_strat( randomtrader )
# print('RandomTrader\n{}'.format(df.tail()))

# df10 = env.env.run_strats( buyhold, Episodes )
# print('BuyHold(Episodes={})\n{}'.format(Episodes, df10.tail()))

# df10 = env.env.run_strats( randomtrader, Episodes )
# print('RandomTrader(Episodes={})\n{}'.format(Episodes, df10.tail()))


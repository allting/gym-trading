import gym

import gym_trading
import pandas as pd
import numpy as np
import trading_env as te

pd.set_option('display.width',500)

env = gym.make('trading-v0')

#env.time_cost_bps = 0
print('Observation size:{}'.format(env.observation_space))

Episodes=5

obs = []

macdsignal = lambda o, e: int(o[0][-1])
reward, df = env.run_strat( macdsignal, render=True )
print('reward:{}, macdsignal\n{}'.format(reward))

buyhold = lambda x,y : 2
reward, df = env.run_strat( buyhold, render=True )
print('reward:{}, BuyHold\n{}'.format(reward))

randomtrader = lambda o,e: e.action_space.sample() # retail trader
reward, df = env.run_strat( randomtrader, render=True )
print('reward:{}, RandomTrader\n{}'.format(reward))

# for _ in range(Episodes):
#     observation = env.reset()
#     env.render()

#     done = False
#     count = 0
#     while not done:
#         action = env.action_space.sample() # random
#         observation, reward, done, info = env.step(action)
#         obs = obs + [observation]
#         #print observation,reward,done,info
#         count += 1

#         env.render()

#         if done:
#             print(reward)
#             print(count)

# df = env.env.sim.to_df()

# print(df.head())
# print(df.tail())

# buyhold = lambda x,y : 2
# reward, df = env.env.run_strat( buyhold )
# print('BuyHold\n{}'.format(df.tail()))

# randomtrader = lambda o,e: e.action_space.sample() # retail trader
# reward, df = env.env.run_strat( randomtrader )
# print('RandomTrader\n{}'.format(df.tail()))

# reward, df10 = env.env.run_strats( buyhold, Episodes )
# print('BuyHold(Episodes={})\n{}'.format(Episodes, df10.tail()))

# reward, df10 = env.env.run_strats( randomtrader, Episodes )
# print('RandomTrader(Episodes={})\n{}'.format(Episodes, df10.tail()))

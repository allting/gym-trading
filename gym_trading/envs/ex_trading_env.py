import gym

import gym_trading
import pandas as pd
import numpy as np
import trading_env as te

pd.set_option('display.width',500)

env = gym.make('trading-v0')

#env.time_cost_bps = 0

Episodes=5

obs = []

for _ in range(Episodes):
    observation = env.reset()
    done = False
    count = 0
    while not done:
        action = env.action_space.sample() # random
        observation, reward, done, info = env.step(action)
        obs = obs + [observation]
        #print observation,reward,done,info
        count += 1
        if done:
            print(reward)
            print(count)

df = env.env.sim.to_df()

print(df.head())
print(df.tail())

buyhold = lambda x,y : 2
df = env.env.run_strat( buyhold )
print('BuyHold\n{}'.format(df.tail()))

randomtrader = lambda o,e: e.action_space.sample() # retail trader
df = env.env.run_strat( randomtrader )
print('RandomTrader\n{}'.format(df.tail()))

df10 = env.env.run_strats( buyhold, Episodes )
print('BuyHold(Episodes={})\n{}'.format(Episodes, df10.tail()))

df10 = env.env.run_strats( randomtrader, Episodes )
print('RandomTrader(Episodes={})\n{}'.format(Episodes, df10.tail()))
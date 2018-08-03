from gym.envs.registration import register

register(
    id='trading-v0',
    entry_point='gym_trading.envs:TradingEnv',
)
#register(
#    id='foo-extrahard-v0',
#    entry_point='gym_foo.envs:FooExtraHardEnv',
#    timestep_limit=1000,
#)

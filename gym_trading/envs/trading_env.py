import gym
from gym import error, spaces, utils
from gym.utils import seeding
from collections import Counter

import quandl
import numpy as np
from numpy import random
import pandas as pd
import logging
import pdb

import tempfile

import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from colour import Color

from talib.abstract import *

log = logging.getLogger(__name__)
log.info('%s logger started.',__name__)


OBS_SIZE = 30

def _sharpe(Returns, freq=252) :
  """Given a set of returns, calculates naive (rfr=0) sharpe """
  return (np.sqrt(freq) * np.mean(Returns))/np.std(Returns)

def _prices2returns(prices):
  px = pd.DataFrame(prices)
  nl = px.shift().fillna(0)
  R = ((px - nl)/nl).fillna(0).replace([np.inf, -np.inf], np.nan).dropna()
  R = np.append( R[0].values, 0)
  return R

class QuandlEnvSrc(object):
  ''' 
  Quandl-based implementation of a TradingEnv's data source.
  
  Pulls data from Quandl, preps for use by TradingEnv and then 
  acts as data provider for each new episode.
  '''

  MinPercentileDays = 100 
  QuandlAuthToken = ""  # not necessary, but can be used if desired
  # Name = "TSE/9994" # https://www.quandl.com/search (use 'Free' filter)
  # Name = 'BITSTAMP/USD'
  Name = 'BITFINEX/BTCUSD'

  def __init__(self, days=252, obs_len=OBS_SIZE, name=Name, auth=QuandlAuthToken, scale=True ):
    self.name = name
    self.auth = auth
    self.days = days+1
    log.info('getting data for %s from quandl...',QuandlEnvSrc.Name)
    df = quandl.get(self.name) if self.auth=='' else quandl.get(self.name, authtoken=self.auth)
    log.info('got data for %s from quandl...',QuandlEnvSrc.Name)

    df.columns = ['High', 'Low', 'Mid', 'Close', 'Bid', 'Ask', 'Volume']
    df_dict = {
      'high':df.High,
      'low':df.Low,
      'close':df.Close,
      'volume':df.Volume
    }

    df['SMA'] = SMA(df_dict, timeperiod=25, price='close')
    df['BB_Upper'], df['BB_Middle'], df['BB_Lower'] = BBANDS(df_dict, 20, 2, 2, prices=['high', 'low', 'close'])
    df['STCH_SlowK'], df['STCH_SlowD'] = STOCH(df_dict, prices=['high', 'low', 'close'])
    df['MACD'], df['MACD_Signal'], df['MACD_Hist'] = MACD(df_dict, fastperiod=12, slowperiod=26, signalperiod=9, prices='close')
    
    macd_analyst = [0]*len(df)
    for i in range(1, len(df)):
      if df['MACD_Signal'][i] < df['MACD'][i] and df['MACD'][i-1] <= df['MACD_Signal'][i-1]:
        macd_analyst[i] = 2 # BUY
      elif df['MACD'][i] < df['MACD_Signal'][i] and df['MACD_Signal'][i-1] <= df['MACD'][i-1]:
        macd_analyst[i] = 1 # Sell
      # else:
      #   macd_analyst.append(0) # Hold
    df['MACD_Analyst'] = macd_analyst

    # df = df[ ~np.isnan(df.Volume)][['Close','Volume']]
    # we calculate returns and percentiles, then kill nans
    # df = df[['Close','Volume']]   
    df.Volume.replace(0,1,inplace=True) # days shouldn't have zero volume..
    df['Return'] = (df.Close-df.Close.shift())/df.Close.shift()
    pctrank = lambda x: pd.Series(x).rank(pct=True).iloc[-1]
    df['ClosePctl'] = df.Close.expanding(self.MinPercentileDays).apply(pctrank)
    df['VolumePctl'] = df.Volume.expanding(self.MinPercentileDays).apply(pctrank)
    df.dropna(axis=0,inplace=True)
    R = df.Return
    dfSMA = df.SMA
    dfBBUpper, dfBBMiddle, dfBBLower = df['BB_Upper'], df['BB_Middle'], df['BB_Lower']
    dfSTCHSlowK, dfSTCHSlowD = df['STCH_SlowK'], df['STCH_SlowD']
    dfMACD, dfMACDSignal, dfMACDHist = df['MACD'], df['MACD_Signal'], df['MACD_Hist']

    if scale:
      mean_values = df.mean(axis=0)
      std_values = df.std(axis=0)
      df = (df - np.array(mean_values))/ np.array(std_values)

    df['Return'] = R # we don't want our returns scaled
    df['SMA'] = dfSMA
    df['BB_Upper'], df['BB_Middle'], df['BB_Lower'] = dfBBUpper, dfBBMiddle, dfBBLower
    df['STCH_SlowK'], df['STCH_SlowD'] = dfSTCHSlowK, dfSTCHSlowD
    df['MACD'], df['MACD_Signal'], df['MACD_Hist'] = dfMACD, dfMACDSignal, dfMACDHist

    self.min_values = df.min(axis=0)
    self.max_values = df.max(axis=0)
    self.data = df
    self.step = 0
    self.obs_len = obs_len

    self.days = len(df)

    
  def reset(self):
    # we want contiguous data
    # self.idx = np.random.randint( low = 0, high=len(self.data.index)-self.days )
    self.idx = self.obs_len
    self.step = self.obs_len

  def _step(self):
    try:
      obs = self.data[self.idx-self.obs_len:self.idx]
    except Exception as e:
      print('error:{}'.format(e))

    self.idx += 1
    self.step += 1
    done = self.step >= self.days
    if done == True:
          print('step:{}, days:{}'.format(self.step, self.days))

    return obs,done

  def _next_day_obs(self):
    try:
      obs = self.data.iloc[self.idx]
    except Exception as e:
      print('error:{}'.format(e))

    return obs

class TradingSim(object) :
  """ Implements core trading simulator for single-instrument univ """

  def __init__(self, steps, obs_len=OBS_SIZE, trading_cost_bps = 1e-3, time_cost_bps = 1e-4):
    # invariant for object life
    self.trading_cost_bps = trading_cost_bps
    self.time_cost_bps    = time_cost_bps
    self.steps            = steps
    self.obs_len          = obs_len
    # change every step
    self.step             = self.obs_len
    self.actions          = np.zeros(self.steps)
    self.navs             = np.ones(self.steps)
    self.mkt_nav         = np.ones(self.steps)
    self.strat_retrns     = np.ones(self.steps)
    self.posns            = np.zeros(self.steps)
    self.costs            = np.zeros(self.steps)
    self.trades           = np.zeros(self.steps)
    self.mkt_retrns       = np.zeros(self.steps)
    self.render_on        = 0
    
  def reset(self):
    self.step = self.obs_len
    self.actions.fill(0)
    self.navs.fill(1)
    self.mkt_nav.fill(1)
    self.strat_retrns.fill(0)
    self.posns.fill(0)
    self.costs.fill(0)
    self.trades.fill(0)
    self.mkt_retrns.fill(0)
    
  def _step(self, action, obs, next_day_obs ):
  # def _step(self, action, retrn ):
    """ Given an action and return for prior period, calculates costs, navs,
        etc and returns the reward and a  summary of the day's activity. """
    retrn = 0
    try:
      retrn = next_day_obs['Return']
    except Exception as e:
      print('Exception - {}'.format(e))

    bod_posn = 0.0 if self.step == 0 else self.posns[self.step-1]
    bod_nav  = 1.0 if self.step == 0 else self.navs[self.step-1]
    mkt_nav  = 1.0 if self.step == 0 else self.mkt_nav[self.step-1]

    reward = 0
    # 0 - flat, 1 - buy, 2 - sell
    if (action == 2 and retrn < 0) or (action == 1 and 0 < retrn):
      reward = 1
    elif (action == 2 and 0 < retrn) or (action == 1 and retrn < 0):
      reward = -1
    else:
      # reward = 1 if 0 < retrn else -1
      reward = ( (bod_posn * retrn) - self.costs[self.step] )

    self.mkt_retrns[self.step] = retrn
    self.actions[self.step] = action
    
    self.posns[self.step] = action - 1     
    self.trades[self.step] = self.posns[self.step] - bod_posn
    
    trade_costs_pct = abs(self.trades[self.step]) * self.trading_cost_bps 
    self.costs[self.step] = trade_costs_pct +  self.time_cost_bps
    # reward = ( (bod_posn * retrn) - self.costs[self.step] )
    self.strat_retrns[self.step] = reward

    if self.step != 0 :
      self.navs[self.step] =  bod_nav * (1 + self.strat_retrns[self.step-1])
      self.mkt_nav[self.step] =  mkt_nav * (1 + self.mkt_retrns[self.step-1])
    
    info = { 'reward': reward, 'nav':self.navs[self.step], 'costs':self.costs[self.step], 'step':self.step }
    self.step += 1      
    return reward, info

  def to_df(self):
    """returns internal state in new dataframe """
    cols = ['action', 'bod_nav', 'mkt_nav','mkt_return','sim_return',
            'position','costs', 'trade' ]
    rets = _prices2returns(self.navs)
    #pdb.set_trace()
    df = pd.DataFrame( {'action':     self.actions, # today's action (from agent)
                          'bod_nav':    self.navs,    # BOD Net Asset Value (NAV)
                          'mkt_nav':  self.mkt_nav, 
                          'mkt_return': self.mkt_retrns,
                          'sim_return': self.strat_retrns,
                          'position':   self.posns,   # EOD position
                          'costs':  self.costs,   # eod costs
                          'trade':  self.trades },# eod trade
                         columns=cols)
    return df

class Render(object):
  def __init__(self, df, obs_len=OBS_SIZE):
        self.render_on = 0
        self.data = df
        self.obs_len = obs_len

  def reset(self):
        self.render_on = 0

  def _plot_trading(self, step, actions):
      close = self.data.Close
      obs_len = self.obs_len 
      price = close[:step]
      obs_price = price[-obs_len:]
      price_x = list(range(len(price)))
      returns = self.data.Return
      features = np.array([returns[:step], price]).T
      feature_len = 2

      features_color = [c.rgb+(0.9,) for c in Color('yellow').range_to(Color('cyan'), feature_len)]

      self.price_plot = self.ax.plot(price_x, price, c=(0, 0.68, 0.95, 0.9),zorder=1)
      self.features_plot = [self.ax3.plot(price_x, features[:, i], 
                                          c=features_color[i])[0] for i in range(feature_len)]
      rect_high = obs_price.max() - obs_price.min()
      self.target_box = self.ax.add_patch(
                          patches.Rectangle(
                          (step-obs_len, obs_price.min()), obs_len, rect_high,
                          label='observation',edgecolor=(0.9, 1, 0.2, 0.8),facecolor=(0.95,1,0.1,0.3),
                          linestyle='-',linewidth=1.5,
                          fill=True)
                          )     # remove background)
      # self.fluc_reward_plot_p = self.ax2.fill_between(price_x, 0, self.reward_fluctuant_arr[:self.step_st+self.obs_len],
      #                                                 where=self.reward_fluctuant_arr[:self.step_st+self.obs_len]>=0, 
      #                                                 facecolor=(1, 0.8, 0, 0.2), edgecolor=(1, 0.8, 0, 0.9), linewidth=0.8)
      # self.fluc_reward_plot_n = self.ax2.fill_between(price_x, 0, self.reward_fluctuant_arr[:self.step_st+self.obs_len],
      #                                                 where=self.reward_fluctuant_arr[:self.step_st+self.obs_len]<=0, 
      #                                                 facecolor=(0, 1, 0.8, 0.2), edgecolor=(0, 1, 0.8, 0.9), linewidth=0.8)
      # self.posi_plot_long = self.ax2.fill_between(price_x, 0, self.posi_arr[:self.step_st+self.obs_len], 
      #                                             where=self.posi_arr[:self.step_st+self.obs_len]>=0, 
      #                                             facecolor=(1, 0.5, 0, 0.2), edgecolor=(1, 0.5, 0, 0.9), linewidth=1)
      # self.posi_plot_short = self.ax2.fill_between(price_x, 0, self.posi_arr[:self.step_st+self.obs_len], 
      #                                               where=self.posi_arr[:self.step_st+self.obs_len]<=0, 
      #                                               facecolor=(0, 0.5, 1, 0.2), edgecolor=(0, 0.5, 1, 0.9), linewidth=1)
      # self.reward_plot_p = self.ax2.fill_between(price_x, 0, 
      #                                             self.reward_arr[:self.step_st+self.obs_len].cumsum(),
      #                                             where=self.reward_arr[:self.step_st+self.obs_len].cumsum()>=0,
      #                                             facecolor=(1, 0, 0, 0.2), edgecolor=(1, 0, 0, 0.9), linewidth=1)
      # self.reward_plot_n = self.ax2.fill_between(price_x, 0, 
      #                                             self.reward_arr[:self.step_st+self.obs_len].cumsum(),
      #                                             where=self.reward_arr[:self.step_st+self.obs_len].cumsum()<=0,
      #                                             facecolor=(0, 1, 0, 0.2), edgecolor=(0, 1, 0, 0.9), linewidth=1)

      # 0 - flat, 1 - buy, 2 - sell
      action = actions[:step]
      trade_x = action.nonzero()[0]
      trade_x_buy = [i for i in trade_x if actions[i]==1]
      trade_x_sell = [i for i in trade_x if actions[i]==2]
      trade_y_buy = [price[i] for i in trade_x_buy]
      trade_y_sell =  [price[i] for i in trade_x_sell]
      trade_color_buy = (1, 0, 0, 0.5) 
      trade_color_sell = (0, 1, 0, 0.5)
      self.trade_plot_buy = self.ax.scatter(x=trade_x_buy, y=trade_y_buy, s=100, marker='^', 
                                            c=trade_color_buy, edgecolors=(1,0,0,0.9), zorder=2)
      self.trade_plot_sell = self.ax.scatter(x=trade_x_sell, y=trade_y_sell, s=100, marker='v', 
                                              c=trade_color_sell, edgecolors=(0,1,0,0.9), zorder=2)


  def render(self, step, actions):
      close = self.data.Close
      price = close[:step]
 
      if self.render_on == 0:
          matplotlib.style.use('dark_background')
          self.render_on = 1

          left, width = 0.1, 0.8
          rect1 = [left, 0.4, width, 0.55]
          rect2 = [left, 0.2, width, 0.2]
          rect3 = [left, 0.05, width, 0.15]

          self.fig = plt.figure(figsize=(15,8))
          # self.fig.suptitle('%s'%self.src.data.iloc[0].date(), fontsize=14, fontweight='bold')
          self.ax = self.fig.add_subplot(1,1,1)
          self.ax = self.fig.add_axes(rect1)  # left, bottom, width, height
          self.ax2 = self.fig.add_axes(rect2, sharex=self.ax)
          self.ax3 = self.fig.add_axes(rect3, sharex=self.ax)
          self.ax.grid(color='gray', linestyle='-', linewidth=0.5)
          self.ax2.grid(color='gray', linestyle='-', linewidth=0.5)
          self.ax3.grid(color='gray', linestyle='-', linewidth=0.5)
          self.features_color = [c.rgb+(0.9,) for c in Color('yellow').range_to(Color('cyan'), 2)]
          #fig, ax = plt.subplots()
          self._plot_trading(step, actions)

          self.ax.set_xlim(0,len(price)+200)
          plt.ion()
          #self.fig.tight_layout()
          plt.show()
          # if save:
          #     self.fig.savefig('fig/%s.png' % str(self.t_index))

      elif self.render_on == 1:
          self.ax.lines.remove(self.price_plot[0])
          [self.ax3.lines.remove(plot) for plot in self.features_plot]
          # self.fluc_reward_plot_p.remove()
          # self.fluc_reward_plot_n.remove()
          self.target_box.remove()
          # self.reward_plot_p.remove()
          # self.reward_plot_n.remove()
          # self.posi_plot_long.remove()
          # self.posi_plot_short.remove()
          self.trade_plot_buy.remove()
          self.trade_plot_sell.remove()

          self._plot_trading(step, actions)

          self.ax.set_xlim(0,len(price)+200)
          # if save:
          #     self.fig.savefig('fig/%s.png' % str(self.t_index))
          plt.pause(0.000001)
        

class TradingEnv(gym.Env):
  """This gym implements a simple trading environment for reinforcement learning.

  The gym provides daily observations based on real market data pulled
  from Quandl on, by default, the SPY etf. An episode is defined as 252
  contiguous days sampled from the overall dataset. Each day is one
  'step' within the gym and for each step, the algo has a choice:

  FLAT (0)
  LONG (1)
  SHORT (2)

  If you trade, you will be charged, by default, 10 BPS of the size of
  your trade. Thus, going from short to long costs twice as much as
  going from short to/from flat. Not trading also has a default cost of
  1 BPS per step. Nobody said it would be easy!

  At the beginning of your episode, you are allocated 1 unit of
  cash. This is your starting Net Asset Value (NAV). If your NAV drops
  to 0, your episode is over and you lose. If your NAV hits 2.0, then
  you win.

  The trading env will track a buy-and-hold strategy which will act as
  the benchmark for the game.

  """
  metadata = {'render.modes': ['human']}

  def __init__(self):
    self.days = 252
    self.src = QuandlEnvSrc(days=self.days)
    self.sim = TradingSim(steps=self.src.days, trading_cost_bps=1e-3,
                          time_cost_bps=1e-4)
    self.chart = Render(df=self.src.data)

    self.action_space = spaces.Discrete( 3 )
    # self.observation_space= spaces.Box( self.src.min_values,
    #                                     self.src.max_values)
    columns = len(self.src.data.columns)
    self.observation_space= spaces.Box( 0,
                                        self.src.days,
                                        shape=(OBS_SIZE, columns))
    self._reset()

  def _configure(self, display=None):
    self.display = display

  def _seed(self, seed=None):
    self.np_random, seed = seeding.np_random(seed)
    return [seed]

  def _step(self, action):
    assert self.action_space.contains(action), "%r (%s) invalid"%(action, type(action))
    observation, done = self.src._step()
    # Close    Volume     Return  ClosePctl  VolumePctl
    # yret = observation[:,2]
    next_day_obs = pd.DataFrame()
    if not done:
      next_day_obs = self.src._next_day_obs()

    reward, info = self.sim._step( action, observation, next_day_obs )
      
    #info = { 'pnl': daypnl, 'nav':self.nav, 'costs':costs }

    return observation.values, reward, done, info
  
  def _reset(self):
    self.src.reset()
    self.sim.reset()
    self.chart.reset()
    obs, _ = self.src._step()
    return obs.values
    
  # def _render(self, mode='human', close=False):
  #   #... TODO
  #   pass

  # some convenience functions:
  
  def run_strat(self,  strategy, return_df=True):
    """run provided strategy, returns dataframe with all steps"""
    observation = self._reset()
    done = False
    while not done:
      action = strategy( observation, self ) # call strategy
      observation, reward, done, info = self.step(action)

    return self.sim.to_df() if return_df else None
      
  def run_strats( self, strategy, episodes=1, write_log=False, return_df=True):
    """ run provided strategy the specified # of times, possibly
        writing a log and possibly returning a dataframe summarizing activity.
    
        Note that writing the log is expensive and returning the df is moreso.  
        For training purposes, you might not want to set both.
    """
    need_df = True
    logfile = None
    if write_log:
      logfile = tempfile.NamedTemporaryFile(delete=False)
      log.info('writing log to %s',logfile.name)
      need_df = write_log or return_df

    alldf = None
        
    for i in range(episodes):
      df = self.run_strat(strategy, return_df=need_df)
      if write_log:
        df.to_csv(logfile, mode='a')
      if return_df:
        alldf = df if alldf is None else pd.concat([alldf,df], axis=0)
            
    return alldf

  def _render(self, mode='human', close=False):
    self.chart.render(self.src.step, self.sim.actions)  

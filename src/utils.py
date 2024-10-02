import numpy as np
import matplotlib.pyplot as plt
from arch import arch_model
import csv


def quantile_loss(x, reference, quantile):
    if x >= reference:
        return (1 - quantile) * (x - reference)
    else:
        return quantile * (reference - x)
    

# GARCH model and score function used in the time series experiment. The following code is taken from https://github.com/ProgBelarus/MultiValidPrediction/blob/main/experiments/Time%20Series%20-%20Stock%20Returns.ipynb

def garch_scores(returns, volatility, lookback=100, offset=10, score_type='regr', score_unit_interval=True, norm_const=-1):
  # initialize prediction model
  garch_model = arch_model(returns, vol='Garch', p=1, q=1)

  predictions = [0 for i in range(offset)]
  scores = [0 for i in range(offset)]

  for t in range(offset, len(volatility)):
      # current window indices
      left_ind = max(0, t-lookback)
      right_ind = t

      # compute score
      variance_pred_array = garch_model.fit(first_obs=left_ind, last_obs=right_ind, disp='off', show_warning=False).forecast(reindex=False).variance
      varNext = variance_pred_array.iloc[0]['h.1'] #['h.1'][lookback-1]
      score = score_fn(actual=volatility[t], pred=varNext, score_type=score_type, unit_interval_norm=score_unit_interval, divide_by_const=(norm_const != -1), norm_const=norm_const)

      # update arrays with data
      scores.append(score)
      predictions.append(varNext)

  return scores, predictions


def score_fn(actual, pred, score_type, unit_interval_norm=False, divide_by_const=False, norm_const=1000):
  # what kind of score?
  if score_type=='regr':
    scr = abs(actual-pred)
  if score_type=='regr_normalized':
    scr = abs(actual-pred)/pred
  
  # normalize score into [0, 1]
  if unit_interval_norm: 
    if divide_by_const:
      scr /= norm_const
    else:
      scr = scr/(1+scr)
  return scr


# Preprocessing time series data

def stock_history(name, datefrom='01/03/00', dateto='12/31/20', ret_scale=100):
  file = open('./Data/' + name + '.csv')
  csv_file = csv.reader(file)
  header= next(csv_file)

  rows = []
  for row in csv_file:
          rows.append(row)

  rows = np.array(rows)

  dates = np.array(rows[:, 0])
  open_prices = np.array([float(price) for price in rows[:, 1]])

  begin = np.where(dates==datefrom)[0][0]

  end = np.where(dates==dateto)[0][0]

  prices = open_prices[end:begin][::-1]

  T = len(prices)

  returns = [(prices[1]/prices[0]-1)]
  for t in range(1, T):
    returns.append(prices[t]/prices[t-1] - 1)
  
  returns = [ret * ret_scale for ret in returns] # scale returns

  volatility = [ret**2 for ret in returns]

  f, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True)
  ax1.plot(prices)
  ax1.set_ylabel('prices')
  ax2.plot(returns)
  ax2.set_ylabel('returns')
  ax3.plot(volatility)
  ax3.set_ylabel('volatility')
  f.suptitle(name + ' Historical Data')
  
  return T, prices, returns, volatility

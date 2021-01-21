# -*- coding: utf-8 -*-

import csv
import numpy as np
import pandas as pd

HIGH_KEY = ' High'
LOW_KEY = ' Low'
CLOSE_KEY = ' Close/Last'
DATE_KEY = 'Date'

def EMA(n, p):
  """
  Exponential Moving Average.
  Arguments:
      n - time period
      p - closing prices

  Returns:
      emas - calculated EMAs for the n-day time period
  """
  p_len = len(p)
  a = 2 / (n + 1)

  emas = []
  emas.append(np.average(p[p_len-n:p_len])) # SMA as first value
              
  for i in range(n, p_len):
    ema = a*p[p_len-i-1] + (1-a)*emas[0]
    emas.insert(0, ema)

  return emas

def RoR (pt, pt_):
  """
  Logarithmic Rate of Return.
  Arguments:
      pt - price for current day
      pt_ - price from n-th previous day (e.g. form previous day for 1-day logarithmic RoR)

  Returns:
      calculated logarithmic RoR
  """
  return np.log(pt/pt_)


def grad_price_trend(p):
  """
  The gradient of the n-day price trend, where n is the length of p.
  Arguments:
      p - prices for n-day time period

  Returns:
      n-day gradient
  """
  n = len(p)
  n_avg = np.average(range(1, n+1))
  p_avg = np.average(p)

  numerator = 0
  denominator = 0
  for i in range(1, n+1):
    numerator += (i - n_avg)*(p[p.index[0] + i-1] - p_avg)
    denominator += (i - n_avg)**2

  return numerator/denominator

def RSI(n, p):
  """
  Relative Strength Index
  Arguments:
      n - time period for EMA calculation
      p - closing prices
  Returns:
      array of calculated RSIs
  """
  U = [] # gain
  D = [] # loss
  for i in range(len(p) - 1):
    if p[i] == p[i+1]:
      U.append(0); D.append(0)
    else:
      if p[i] > p[i+1]:
        U.append (p[i] - p[i+1]); D.append(0)
      else:
        U.append(0); D.append(p[i+1] - p[i])

  if not D:
    return 100
  if not U:
    return 0

  U_ema = EMA(n,U)
  D_ema = EMA(n,D)

  RS = np.divide(U_ema,D_ema)

  return 100 - 100/(1 + RS)

def MACD(p):
  """
  Moving Average Convergence / Divergence as difference between 12-day EMA and 26-day EMA
  Arguments:
      p - closing prices
  Returns:
      array of calculated MACDs
  """
  ema26 = EMA(26, p)
  ema12 = EMA(12, p)[0:len(ema26)] 

  return np.subtract(ema12,ema26)

def CCI(p, h, l):
  """
  Commodity Channel Index for n-say time period
  Arguments:
      p - closing prices for the n-day time period
      h - highest day prices for the n-day time period
      l - lowest day prices for the n-day time period
  Returns:
     result - calculated CCI
  """
  typical_price = np.add(np.add(p, h), l)/3
  MA = np.average(typical_price)
  mean_deviation = np.average(np.abs(typical_price - MA))
  result = (typical_price[typical_price.index[0]] - MA)/(0.015 * mean_deviation)
  return result

def calculate_features(d):
  """
  Calculate 16 features based on historical stock data
  Arguments:
      d - DateFrame containing closing, lowest and highest prices
  Returns:
     DateFrame containing calculated features
  """
  features = []
  for i in range(17):
    features.append([])
  print(features)

  # Relative Strength Index for 14 days
  features[12] = RSI(14, d.get(CLOSE_KEY))[0:len(d.get(CLOSE_KEY)) - 26]
  features[13] = MACD(d.get(CLOSE_KEY))[0:len(d.get(CLOSE_KEY)) - 26]

  for i in range(len(d.get(CLOSE_KEY)) - 26):
    features[1].append(RoR (d.get(CLOSE_KEY)[i], d.get(CLOSE_KEY)[i+1]))
    features[2].append(RoR (d.get(CLOSE_KEY)[i+1], d.get(CLOSE_KEY)[i+2]))
    features[3].append(RoR (d.get(CLOSE_KEY)[i+2], d.get(CLOSE_KEY)[i+3]))
    features[4].append(RoR (d.get(CLOSE_KEY)[i+3], d.get(CLOSE_KEY)[i+4]))

    features[5].append(RoR (d.get(CLOSE_KEY)[i], d.get(CLOSE_KEY)[i+2]))
    features[6].append(RoR (d.get(CLOSE_KEY)[i+1], d.get(CLOSE_KEY)[i+3]))

    # gradient of 5-day price trend
    features[7].append(grad_price_trend(d.get(CLOSE_KEY).take(range(i,i+5))))
    features[8].append(grad_price_trend(d.get(CLOSE_KEY).take(range(i+5,i+10))))
    # gradient of 10-day price trend
    features[9].append(grad_price_trend(d.get(CLOSE_KEY).take(range(i,i+10))))

    features[10].append(features[1][i] - features[2][i])
    features[11].append(features[1][i] - features[3][i])

    # f12 = RSI[i]
    # f13 = MACD[i]

    features[14].append(d.get(CLOSE_KEY)[i] - np.average(d.get(CLOSE_KEY).take(range(i+1,i+13))))

    # 14-day rate of change 
    features[15].append((d.get(CLOSE_KEY)[i] - d.get(CLOSE_KEY)[i+14])/d.get(CLOSE_KEY)[i+14])

    features[16].append(CCI(d.get(CLOSE_KEY).take(range(i,i+20)), d.get(HIGH_KEY).take(range(i,i+20)), d.get(LOW_KEY).take(range(i,i+20))))

  result = {
      DATE_KEY: d.get(DATE_KEY).take(range(0,len(d.get(CLOSE_KEY)) - 26)),
      CLOSE_KEY: d.get(CLOSE_KEY).take(range(0,len(d.get(CLOSE_KEY)) - 26)),
      HIGH_KEY: d.get(HIGH_KEY).take(range(0,len(d.get(CLOSE_KEY)) - 26)),
      LOW_KEY: d.get(LOW_KEY).take(range(0,len(d.get(CLOSE_KEY)) - 26))
  }

  for i in range(1,17):
    result['f'+str(i)] = features[i]
  print(type(result))
  print(pd.DataFrame(data=result))
  return pd.DataFrame(data=result)

def read_and_extract_features(input, output):
  df = pd.read_csv (input)
  df[CLOSE_KEY] = df[CLOSE_KEY].copy().apply(lambda x: float(x[2:]))
  df[HIGH_KEY] = df[HIGH_KEY].copy().apply(lambda x: float(x[2:]))
  df[LOW_KEY] = df[LOW_KEY].copy().apply(lambda x: float(x[2:]))
  print(df)

  features = calculate_features(df)
  features.to_csv(output)

if __name__ == "__main__":
   read_and_extract_features('BIDU-5y.csv', 'features-BIDU-5y.csv')
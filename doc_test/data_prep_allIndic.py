

%reset -f
import pickle
import pandas as pd
import numpy as np
import ta
import scipy.signal as scipy
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' 

def signal_filtfilt(signal, step=0.2):
    B, A = scipy.butter(3, step, output='ba')
    my_signal = scipy.filtfilt(B, A, signal)
    return my_signal

extensionFile = "_inf24_15m.pkl"
# extensionFile = "_24_15m.pkl"
tabSymbol = ['BTC', 'ETH', 'BNB', 'XRP', 'ADA', 'SOL', 'DOT',  'LTC', 'LINK', 'XLM']
# tabSymbol = ['BTC']

for aa in range(0, len(tabSymbol)):  
    symbol = tabSymbol[aa] + 'USDT'
    strFile = "C:\\ml/data" + symbol + "PreProc_" + extensionFile
    df = pickle.load(open(strFile,'rb'))
    
    print(strFile)
    
    df['RSI5'] = ta.momentum.RSIIndicator(close=df['close'], window=5, fillna=True).rsi()
    df['RSI9'] = ta.momentum.RSIIndicator(close=df['close'], window=9, fillna=True).rsi()
    df['RSI14'] = ta.momentum.RSIIndicator(close=df['close'], window=14, fillna=True).rsi()
    
    df['CCI9'] = ta.trend.CCIIndicator(df['high'], df['low'], df['close'], window=9, constant= 0.015, fillna= True).cci()
    df['CCI13'] = ta.trend.CCIIndicator(df['high'], df['low'], df['close'], window=13, constant= 0.015, fillna= True).cci()
    df['CCI20'] = ta.trend.CCIIndicator(df['high'], df['low'], df['close'], window=20, constant= 0.015, fillna= True).cci()
    
    df['MACD13'] = ta.trend.MACD(close=df['close'], window_slow = 13, window_fast = 6, window_sign = 4, fillna = True).macd_diff()
    df['MACD26'] = ta.trend.MACD(close=df['close'], window_slow = 26, window_fast = 12, window_sign = 9, fillna = True).macd_diff()
    df['MACD40'] = ta.trend.MACD(close=df['close'], window_slow = 40, window_fast = 18, window_sign = 13, fillna = True).macd_diff()
    
    df.fillna(0, inplace=True)    
    
    df['FL_RSI5_15'] = signal_filtfilt(np.array(df['RSI5']), step=0.15)
    df['FL_RSI9_15'] = signal_filtfilt(np.array(df['RSI9']), step=0.15)
    df['FL_RSI14_15'] = signal_filtfilt(np.array(df['RSI14']), step=0.15)
    df['FL_RSI5_20'] = signal_filtfilt(np.array(df['RSI5']), step=0.20)
    df['FL_RSI9_20'] = signal_filtfilt(np.array(df['RSI9']), step=0.20)
    df['FL_RSI14_20'] = signal_filtfilt(np.array(df['RSI14']), step=0.20)
    
    df['FL_CCI9_15'] = signal_filtfilt(np.array(df['CCI9']), step=0.15)
    df['FL_CCI13_15'] = signal_filtfilt(np.array(df['CCI13']), step=0.15)
    df['FL_CCI20_15'] = signal_filtfilt(np.array(df['CCI20']), step=0.15)
    df['FL_CCI9_20'] = signal_filtfilt(np.array(df['CCI9']), step=0.20)
    df['FL_CCI13_20'] = signal_filtfilt(np.array(df['CCI13']), step=0.20)
    df['FL_CCI20_20'] = signal_filtfilt(np.array(df['CCI20']), step=0.20)
    
    df['FL_MACD13_15'] = signal_filtfilt(np.array(df['MACD13']), step=0.15)
    df['FL_MACD26_15'] = signal_filtfilt(np.array(df['MACD26']), step=0.15)
    df['FL_MACD40_15'] = signal_filtfilt(np.array(df['MACD40']), step=0.15)
    df['FL_MACD13_20'] = signal_filtfilt(np.array(df['MACD13']), step=0.20)
    df['FL_MACD26_20'] = signal_filtfilt(np.array(df['MACD26']), step=0.20)
    df['FL_MACD40_20'] = signal_filtfilt(np.array(df['MACD40']), step=0.20)
    
    
    df.fillna(0, inplace=True)
    df = df.reset_index(drop=True)

    df.set_index('timestamp', inplace=True)
    # Setting 15min frequency
    df.index.freq = '15min'
    df = df.reset_index(drop=True)
    
    strFile = "C:\\ml/CLOSE_FIL/data" + symbol + "Train" + extensionFile
    pickle.dump(df, open(strFile, "wb"))
    

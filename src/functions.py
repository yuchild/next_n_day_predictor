from pandas import read_pickle as rp
from yfinance import Ticker
import numpy as np
from csv import reader
from os import path
from datetime import date, datetime
from math import ceil

import warnings
warnings.filterwarnings('ignore')

def read_symbols_csv():

    file_path = path.join("static", "symbols.csv")
   
    result = []
    
    # Check if the file exists
    if not path.exists(file_path):
        print(f"The file {file_path} does not exist.")
        return result
    
    with open(file_path, 'r') as file:
        csv_reader = reader(file)
        
        for row in csv_reader:
            # Assuming the CSV file has two columns: symbol and date
            if len(row) == 2:
                symbol, date = row
                result.append((symbol, date))
            else:
                print(f"Skipping invalid row: {row}")
    
    return result


def current_time():
    
    today = date.today()
    now = datetime.now().strftime("%H:%M:%S")
    
    return f'{today} {now}'
    

def download_tables(stock_list = read_symbols_csv()):
    
    start_time = current_time()
    
    ctn = 0
    
    for item in stock_list:
        
        stock = Ticker(item[0])
        
        #get max 1 day data
        stock_1d_df = stock.history(start = item[1],  # may not be necessary as period='max'
                                    end = None,
                                    interval = '1d',  # time spacing interval
                                    period='max',  # historical period, can adjust start and end
                                    auto_adjust=False, # new as of 1/23/24
                                   )
        stock_1d_df.to_pickle(f'./data/{item[0]}_1d_df.pkl')
        
        #get max 1 hour data
        stock_1h_df = stock.history(interval = '1h',  # time spacing interval
                                    period='60d',  # historical period, can use start and end
                                    auto_adjust=False, # new as of 1/23/24
                                   )
        stock_1h_df.to_pickle(f'./data/{item[0]}_1h_df.pkl')
        
        ctn += 1
    
    end_time = current_time()
        
    return print(f'Start time: {start_time}\nDownloaded {ctn} max daily and hourly stock data\nEnd Time: {end_time}')


def candle_parts_pcts(o, c, h, l):
    full = h - l
    body = abs(o - c)
    if o > c:
        top_wick = h - o
        bottom_wick = c - l
    else:
        top_wick = h - c
        bottom_wick = o - l
    return top_wick / full, body / full, bottom_wick / full


def gap_up_down_pct(o, pc, ph, pl):
    if o > pc:
        return (o - pc) / (ph - pl)
    elif o == pc:
        return 0
    else:
        return (pc - o) / (ph - pl)
    

def load_transform_tables(stock_list = read_symbols_csv()):
    
    start_time = current_time()
    
    ctn = 0
    
    for item in stock_list:
        
        #get max 1 day data
        stock_1d_df = rp(f'./data/{item[0]}_1d_df.pkl')
        
        #update 1 day table: candle %'s
        stock_1d_df[['pct_top_wick', 'pct_body', 'pct_bottom_wick']] = stock_1d_df.apply(lambda row: candle_parts_pcts(row['open'], row['close'], row['high'],  row['low']), axis=1, result_type='expand')
        
        #update 1 day table: % gap btwn candles relative to previous candle size
        stock_1d_df['pc'] = stock_1d_df['close'].shift(1).copy()
        stock_1d_df['ph'] = stock_1d_df['high'].shift(1).copy()
        stock_1d_df['pl'] = stock_1d_df['low'].shift(1).copy()
        stock_1d_df['pct_gap_up_down'] = stock_1d_df.apply(lambda row: gap_up_down_pct(row['open'], row['pc'], row['ph'], row['pl']), axis=1, result_type='expand')
        
        
                                                          
        
        #get max 1 hour data
        stock_1h_df = stock.history(interval = '1h',  # time spacing interval
                                    period='60d',  # historical period, can use start and end
                                    auto_adjust=False, # new as of 1/23/24
                                   )
        stock_1h_df.to_pickle(f'./data/{item[0]}_1h_df.pkl')
        
        ctn += 1
    
    end_time = current_time()
        
    return print(f'Start time: {start_time}\nDownloaded {ctn} max daily and hourly stock data\nEnd Time: {end_time}')

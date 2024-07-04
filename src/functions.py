from pandas import read_pickle as rp
from yfinance import Ticker
import numpy as np
from csv import reader
from os import path
from datetime import date, datetime
from math import ceil

import warnings
warnings.filterwarnings('ignore')


#########################################
# functions for use to transform tables #
#########################################

# candle parts percentages
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


# previous close and open gap % of pervious candle size
def gap_up_down_pct(o, pc, ph, pl):
    if o > pc:
        return (o - pc) / (ph - pl)
    elif o == pc:
        return 0
    else:
        return (pc - o) / (ph - pl)
    
    
# z-score calculation
def zscore(x, mu, stdev):
    return (x - mu) / stdev


# direction calculation:
def direction(today, tomorrow):
    pct_change = (tomorrow - today) / today
    if pct_change > 0.0075:
        return 1
    elif pct_change < -0.0075:
        return -1
    else:
        return 0
    

###########################################
# functions for use to get ETL/ELT tables #
###########################################

# read curated stock symbols from static file
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


# returns now timestamp
def current_time():
    
    today = date.today()
    now = datetime.now().strftime("%H:%M:%S")
    
    return f'{today} {now}'
    

# download tables from yfinance (yahoo finance)
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


# load downloaded tables, transform for machine learning
def load_transform_tables(stock_list = read_symbols_csv()):
    
    start_time = current_time()
    
    ctn = 0
    
    for item in stock_list:
        ######################
        #transform 1 day data#
        ######################
        stock_1d_df = rp(f'./data/{item[0]}_1d_df.pkl')
        
        #update 1 day table: candle parts %'s
        stock_1d_df[['pct_top_wick', 'pct_body', 'pct_bottom_wick']] = stock_1d_df.apply(lambda row: candle_parts_pcts(row['Open'], row['Close'], row['High'],  row['Low']), axis=1, result_type='expand').copy()
        
        #stdev of adjusted close
        stock_1d_df['top_stdev21'] = stock_1d_df['pct_top_wick'].rolling(window=21).std().copy() 
        stock_1d_df['body_stdev21'] = stock_1d_df['pct_body'].rolling(window=21).std().copy() 
        stock_1d_df['bottom_stdev21'] = stock_1d_df['pct_bottom_wick'].rolling(window=21).std().copy()

        #mean of adjusted close
        stock_1d_df['top_mu21'] = stock_1d_df['pct_top_wick'].rolling(window=21).mean().copy() 
        stock_1d_df['body_mu21'] = stock_1d_df['pct_body'].rolling(window=21).mean().copy() 
        stock_1d_df['bottom_mu21'] = stock_1d_df['pct_bottom_wick'].rolling(window=21).mean().copy()
        
        #z-score of adjusted close
        stock_1d_df['top_z21'] = stock_1d_df.apply(lambda row: zscore(row['pct_top_wick'], row['top_mu21'], row['top_stdev21']), axis=1, result_type='expand').copy()
        stock_1d_df['body_z21'] = stock_1d_df.apply(lambda row: zscore(row['pct_body'], row['body_mu21'], row['body_stdev21']), axis=1, result_type='expand').copy()
        stock_1d_df['bottom_z21'] = stock_1d_df.apply(lambda row: zscore(row['pct_bottom_wick'], row['bottom_mu21'], row['bottom_stdev21']), axis=1, result_type='expand').copy()
        
        #update 1 day table: % gap btwn current open relative to previous candle size
        stock_1d_df['pc'] = stock_1d_df['Close'].shift(1).copy()
        stock_1d_df['ph'] = stock_1d_df['High'].shift(1).copy()
        stock_1d_df['pl'] = stock_1d_df['Low'].shift(1).copy()
        stock_1d_df['pct_gap_up_down'] = stock_1d_df.apply(lambda row: gap_up_down_pct(row['Open'], row['pc'], row['ph'], row['pl']), axis=1, result_type='expand').copy()
        
        #stdev of adjusted close
        stock_1d_df['ac_stdev5'] = stock_1d_df['Adj Close'].rolling(window=5).std().copy() 
        stock_1d_df['ac_stdev8'] = stock_1d_df['Adj Close'].rolling(window=8).std().copy() 
        stock_1d_df['ac_stdev13'] = stock_1d_df['Adj Close'].rolling(window=13).std().copy()

        #mean of adjusted close
        stock_1d_df['ac_mu5'] = stock_1d_df['Adj Close'].rolling(window=5).mean().copy() 
        stock_1d_df['ac_mu8'] = stock_1d_df['Adj Close'].rolling(window=8).mean().copy() 
        stock_1d_df['ac_mu13'] = stock_1d_df['Adj Close'].rolling(window=13).mean().copy()
        
        #z-score of adjusted close
        stock_1d_df['ac_z5'] = stock_1d_df.apply(lambda row: zscore(row['Adj Close'], row['ac_mu5'], row['ac_stdev5']), axis=1, result_type='expand').copy()
        stock_1d_df['ac_z8'] = stock_1d_df.apply(lambda row: zscore(row['Adj Close'], row['ac_mu8'], row['ac_stdev8']), axis=1, result_type='expand').copy()
        stock_1d_df['ac_z13'] = stock_1d_df.apply(lambda row: zscore(row['Adj Close'], row['ac_mu13'], row['ac_stdev13']), axis=1, result_type='expand').copy()
               
        #target column: direction: -1, 0, 1
        stock_1d_df['adj_close_up1'] = stock_1d_df['Adj Close'].shift(-1).copy()
        stock_1d_df['direction'] = stock_1d_df.apply(lambda row: direction(row['Adj Close'], row['adj_close_up1']), axis=1, result_type='expand').copy() 
        
        #save 1d file for model building
        stock_1d_df[['pct_top_wick', 
                     'pct_body', 
                     'pct_bottom_wick',
                     'top_z21',
                     'body_z21',
                     'bottom_z21',
                     'ac_z5',
                     'ac_z8',
                     'ac_z13',
                     'Adj Close',
                     'direction',
                    ]
                   ].to_pickle(f'./models/{item[0]}_1d_model_df.pkl')
        
        ######################
        #transform 1 day data#
        ######################
        stock_1h_df = rp(f'./data/{item[0]}_1h_df.pkl')
        
        #update 1 day table: candle parts %'s
        stock_1h_df[['pct_top_wick', 'pct_body', 'pct_bottom_wick']] = stock_1h_df.apply(lambda row: candle_parts_pcts(row['Open'], row['Close'], row['High'],  row['Low']), axis=1, result_type='expand').copy()
        
        #stdev of adjusted close
        stock_1h_df['top_stdev21'] = stock_1h_df['pct_top_wick'].rolling(window=21).std().copy() 
        stock_1h_df['body_stdev21'] = stock_1h_df['pct_body'].rolling(window=21).std().copy() 
        stock_1h_df['bottom_stdev21'] = stock_1h_df['pct_bottom_wick'].rolling(window=21).std().copy()

        #mean of adjusted close
        stock_1h_df['top_mu21'] = stock_1h_df['pct_top_wick'].rolling(window=21).mean().copy() 
        stock_1h_df['body_mu21'] = stock_1h_df['pct_body'].rolling(window=21).mean().copy() 
        stock_1h_df['bottom_mu21'] = stock_1h_df['pct_bottom_wick'].rolling(window=21).mean().copy()
        
        #z-score of adjusted close
        stock_1h_df['top_z21'] = stock_1h_df.apply(lambda row: zscore(row['pct_top_wick'], row['top_mu21'], row['top_stdev21']), axis=1, result_type='expand').copy()
        stock_1h_df['body_z21'] = stock_1h_df.apply(lambda row: zscore(row['pct_body'], row['body_mu21'], row['body_stdev21']), axis=1, result_type='expand').copy()
        stock_1h_df['bottom_z21'] = stock_1h_df.apply(lambda row: zscore(row['pct_bottom_wick'], row['bottom_mu21'], row['bottom_stdev21']), axis=1, result_type='expand').copy()
        
        #update 1 day table: % gap btwn current open relative to previous candle size
        stock_1h_df['pc'] = stock_1h_df['Close'].shift(1).copy()
        stock_1h_df['ph'] = stock_1h_df['High'].shift(1).copy()
        stock_1h_df['pl'] = stock_1h_df['Low'].shift(1).copy()
        stock_1h_df['pct_gap_up_down'] = stock_1h_df.apply(lambda row: gap_up_down_pct(row['Open'], row['pc'], row['ph'], row['pl']), axis=1, result_type='expand').copy()
        
        #stdev of adjusted close
        stock_1h_df['ac_stdev5'] = stock_1h_df['Adj Close'].rolling(window=5).std().copy() 
        stock_1h_df['ac_stdev8'] = stock_1h_df['Adj Close'].rolling(window=8).std().copy() 
        stock_1h_df['ac_stdev13'] = stock_1h_df['Adj Close'].rolling(window=13).std().copy()

        #mean of adjusted close
        stock_1h_df['ac_mu5'] = stock_1h_df['Adj Close'].rolling(window=5).mean().copy() 
        stock_1h_df['ac_mu8'] = stock_1h_df['Adj Close'].rolling(window=8).mean().copy() 
        stock_1h_df['ac_mu13'] = stock_1h_df['Adj Close'].rolling(window=13).mean().copy()
        
        #z-score of adjusted close
        stock_1h_df['ac_z5'] = stock_1h_df.apply(lambda row: zscore(row['Adj Close'], row['ac_mu5'], row['ac_stdev5']), axis=1, result_type='expand').copy()
        stock_1h_df['ac_z8'] = stock_1h_df.apply(lambda row: zscore(row['Adj Close'], row['ac_mu8'], row['ac_stdev8']), axis=1, result_type='expand').copy()
        stock_1h_df['ac_z13'] = stock_1h_df.apply(lambda row: zscore(row['Adj Close'], row['ac_mu13'], row['ac_stdev13']), axis=1, result_type='expand').copy()
               
        #target column: direction: -1, 0, 1
        stock_1h_df['adj_close_up1'] = stock_1h_df['Adj Close'].shift(-1).copy()
        stock_1h_df['direction'] = stock_1h_df.apply(lambda row: direction(row['Adj Close'], row['adj_close_up1']), axis=1, result_type='expand').copy() 
        
        #save 1h file for model building
        stock_1h_df[['pct_top_wick', 
                     'pct_body', 
                     'pct_bottom_wick',
                     'top_z21',
                     'body_z21',
                     'bottom_z21',
                     'ac_z5',
                     'ac_z8',
                     'ac_z13',
                     'Adj Close',
                     'direction',
                    ]
                   ].to_pickle(f'./models/{item[0]}_1h_model_df.pkl')
        
        ctn += 1
    
    end_time = current_time()
        
    return print(f'Start time: {start_time}\nDownloaded {ctn} max daily and hourly stock data\nEnd Time: {end_time}')

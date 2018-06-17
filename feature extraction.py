
# coding: utf-8

# In[ ]:


## Load public awareness and economic features for model##
def public_awareness_data(file):
    data = pd.read_excel(file)
    gdeltd = data[['SQLDATE', 'AvgTone']].copy()
    grouped = gdeltd.groupby(['SQLDATE'])
    group_size = grouped.size()
    group_tone = grouped.aggregate(np.mean)
    group_tone['group_size'] = group_size
    group_tone1 = group_tone.iloc[::-1]
    group_tone1 = group_tone1.reset_index()
    group_tone1['SQLDATE'] = pd.to_datetime(group_tone1['SQLDATE'], format='%Y%m%d')
    #group_tone1.to_csv('public_awareness_data.csv')
    return group_tone1

def economic_data(in_file):
    data = pd.read_csv(in_file, sep='\t', header=None)
    data.columns = ['Date','Open','High','Low','Close','Volume','Market Cap']
    BTC_econ_data = data
    BTC_econ_data['Date'] = pd.to_datetime(BTC_econ_data['Date'])
    
    BTC_econ_data['Open'] = BTC_econ_data['Open'].str.replace(',', '')
    BTC_econ_data['Open']= pd.to_numeric(BTC_econ_data['Open'])

    BTC_econ_data['High'] = BTC_econ_data['High'].str.replace(',', '')
    BTC_econ_data['High']= pd.to_numeric(BTC_econ_data['High'])

    BTC_econ_data['Low'] = BTC_econ_data['Low'].str.replace(',', '')
    BTC_econ_data['Low']= pd.to_numeric(BTC_econ_data['Low'])

    BTC_econ_data['Close'] = BTC_econ_data['Close'].str.replace(',', '')
    BTC_econ_data['Close']= pd.to_numeric(BTC_econ_data['Close'])

    BTC_econ_data['Volume'] = BTC_econ_data['Volume'].str.replace(',', '')
    BTC_econ_data['Volume']= pd.to_numeric(BTC_econ_data['Volume'])

    BTC_econ_data['Market Cap'] = BTC_econ_data['Market Cap'].str.replace(',', '')
    BTC_econ_data['Market Cap'] = pd.to_numeric(BTC_econ_data['Market Cap'])

    return BTC_econ_data

def dji_data(in_file):
    dji = pd.read_csv(in_file, header=0)
    dji['Date'] = pd.to_datetime(dji['Date'])
    dji['DJI_Close'] = dji['Close']
    dji = dji[['Date','DJI_Close']]
    dji = dji.iloc[::-1]
    return dji

def google_adjustment_factor(d):
    #where d is a numpy array
    week = 0
    for i in range(1135):
        d[i][3] = d[week][2]/d[week][1]
        if (i+1)% 7 == 0:
            week += 7
    return d  

def google_normalized_search_freq(d):
    for i in range(1135):
        d[i][4] = d[i][1] * d[i][3]
    return d

def google_trends_data(daily_file, weekly_file, currency):
    #Check manually that CSV column names do not clash
    
    daily = pd.read_csv(daily_file, sep=',', header = 1)
    weekly = pd.read_csv(weekly_file, sep=',', header=1)
    daily['Day'] = pd.to_datetime(daily['Day'])
    weekly['Week'] = pd.to_datetime(weekly['Week'])
    #join files on date
    data = daily.set_index('Day').join(weekly.set_index('Week'))
    data = data.reset_index()
    # add adjustment factor and normalized search frequency columns
    data['adjFAC'] = ''
    data['NSF'] = ''
    #handle non numeric data
    data = data.replace(['<1'], [0])
    data.iloc[:, 2] = pd.to_numeric(data.iloc[:, 2])
    # turn to array to manipulate data
    d = np.asarray(data)
    adj_f_d = google_adjustment_factor(d)
    NSF_d = google_normalized_search_freq(adj_f_d)
    final = pd.DataFrame(NSF_d, columns = ['Day',currency +' SF',currency +' WSF','adjFAC','NSF'])
    return final
 
def gold_price(infile):
    gold = pd.read_csv(infile, sep=',')
    gold['Gold_Close'] = gold['Close']
    gold = gold[['Date', 'Gold_Close']]
    gold['Date'] = pd.to_datetime(gold['Date'])
    gold = gold.iloc[::-1]
    return gold

def sse_price(infile):
    sse = pd.read_csv(infile, header=0)
    sse['Date'] = pd.to_datetime(sse['Date'])
    sse['SSE_Close'] = sse['Close']
    sse = sse[['Date','SSE_Close']]
    sse = sse.iloc[::-1]
    return sse

def compile_all(econ_data, public_awareness, dji, google_trends, gold, sse, currency):
    result = econ_data.set_index('Date').join(public_awareness.set_index('SQLDATE'))
    result1 = result.join(dji.set_index('Date'))
    #back fill DJI closing price for weekends
    result1['DJI_Close'] = result1['DJI_Close'].fillna(method='bfill')
    result2 = result1.join(google_trends.set_index('Day'))
    result3 = result2.join(gold.set_index('Date'))
    result3['Gold_Close'] = result3['Gold_Close'].fillna(method='bfill')
    result4 = result3.join(sse.set_index('Date'))
    result4['SSE_Close']= result4['SSE_Close'].fillna(method='bfill')
    #result4.to_csv('_training_data ' + currency)
    return result4def public_awareness_data(file):
    data = pd.read_excel(file)
    gdeltd = data[['SQLDATE', 'AvgTone']].copy()
    grouped = gdeltd.groupby(['SQLDATE'])
    group_size = grouped.size()
    group_tone = grouped.aggregate(np.mean)
    group_tone['group_size'] = group_size
    group_tone1 = group_tone.iloc[::-1]
    group_tone1 = group_tone1.reset_index()
    group_tone1['SQLDATE'] = pd.to_datetime(group_tone1['SQLDATE'], format='%Y%m%d')
    #group_tone1.to_csv('public_awareness_data.csv')
    return group_tone1

def economic_data(in_file):
    data = pd.read_csv(in_file, sep='\t', header=None)
    data.columns = ['Date','Open','High','Low','Close','Volume','Market Cap']
    BTC_econ_data = data
    BTC_econ_data['Date'] = pd.to_datetime(BTC_econ_data['Date'])
    
    BTC_econ_data['Open'] = BTC_econ_data['Open'].str.replace(',', '')
    BTC_econ_data['Open']= pd.to_numeric(BTC_econ_data['Open'])

    BTC_econ_data['High'] = BTC_econ_data['High'].str.replace(',', '')
    BTC_econ_data['High']= pd.to_numeric(BTC_econ_data['High'])

    BTC_econ_data['Low'] = BTC_econ_data['Low'].str.replace(',', '')
    BTC_econ_data['Low']= pd.to_numeric(BTC_econ_data['Low'])

    BTC_econ_data['Close'] = BTC_econ_data['Close'].str.replace(',', '')
    BTC_econ_data['Close']= pd.to_numeric(BTC_econ_data['Close'])

    BTC_econ_data['Volume'] = BTC_econ_data['Volume'].str.replace(',', '')
    BTC_econ_data['Volume']= pd.to_numeric(BTC_econ_data['Volume'])

    BTC_econ_data['Market Cap'] = BTC_econ_data['Market Cap'].str.replace(',', '')
    BTC_econ_data['Market Cap'] = pd.to_numeric(BTC_econ_data['Market Cap'])

    return BTC_econ_data

def dji_data(in_file):
    dji = pd.read_csv(in_file, header=0)
    dji['Date'] = pd.to_datetime(dji['Date'])
    dji['DJI_Close'] = dji['Close']
    dji = dji[['Date','DJI_Close']]
    dji = dji.iloc[::-1]
    return dji

def google_adjustment_factor(d):
    #where d is a numpy array
    week = 0
    for i in range(1135):
        d[i][3] = d[week][2]/d[week][1]
        if (i+1)% 7 == 0:
            week += 7
    return d  

def google_normalized_search_freq(d):
    for i in range(1135):
        d[i][4] = d[i][1] * d[i][3]
    return d

def google_trends_data(daily_file, weekly_file, currency):
    #Check manually that CSV column names do not clash
    
    daily = pd.read_csv(daily_file, sep=',', header = 1)
    weekly = pd.read_csv(weekly_file, sep=',', header=1)
    daily['Day'] = pd.to_datetime(daily['Day'])
    weekly['Week'] = pd.to_datetime(weekly['Week'])
    #join files on date
    data = daily.set_index('Day').join(weekly.set_index('Week'))
    data = data.reset_index()
    # add adjustment factor and normalized search frequency columns
    data['adjFAC'] = ''
    data['NSF'] = ''
    #handle non numeric data
    data = data.replace(['<1'], [0])
    data.iloc[:, 2] = pd.to_numeric(data.iloc[:, 2])
    # turn to array to manipulate data
    d = np.asarray(data)
    adj_f_d = google_adjustment_factor(d)
    NSF_d = google_normalized_search_freq(adj_f_d)
    final = pd.DataFrame(NSF_d, columns = ['Day',currency +' SF',currency +' WSF','adjFAC','NSF'])
    return final
 
def gold_price(infile):
    gold = pd.read_csv(infile, sep=',')
    gold['Gold_Close'] = gold['Close']
    gold = gold[['Date', 'Gold_Close']]
    gold['Date'] = pd.to_datetime(gold['Date'])
    gold = gold.iloc[::-1]
    return gold

def sse_price(infile):
    sse = pd.read_csv(infile, header=0)
    sse['Date'] = pd.to_datetime(sse['Date'])
    sse['SSE_Close'] = sse['Close']
    sse = sse[['Date','SSE_Close']]
    sse = sse.iloc[::-1]
    return sse

def compile_all(econ_data, public_awareness, dji, google_trends, gold, sse, currency):
    result = econ_data.set_index('Date').join(public_awareness.set_index('SQLDATE'))
    result1 = result.join(dji.set_index('Date'))
    #back fill DJI closing price for weekends
    result1['DJI_Close'] = result1['DJI_Close'].fillna(method='bfill')
    result2 = result1.join(google_trends.set_index('Day'))
    result3 = result2.join(gold.set_index('Date'))
    result3['Gold_Close'] = result3['Gold_Close'].fillna(method='bfill')
    result4 = result3.join(sse.set_index('Date'))
    result4['SSE_Close']= result4['SSE_Close'].fillna(method='bfill')
    #result4.to_csv('_training_data ' + currency)
    return result4

## Additional Block chain technical features if required ##
def chain_info(daily_t, diff, fee_percentage, hash_r, block_t ):
    # load block chain info and compile dataframe on date index
    daily_trans = pd.read_csv(daily_t, header = 0)
    daily_trans['Date'] = pd.to_datetime(daily_trans['Date'])
    daily_trans = daily_trans.set_index('Date')
    difficulty = pd.read_csv(diff, header = 0)
    difficulty['Date'] = pd.to_datetime(difficulty['Date'])
    difficulty = difficulty.set_index('Date')
    fee_per = pd.read_csv(fee_percentage, header = 0)
    fee_per['Date'] = pd.to_datetime(fee_per['Date'])
    fee_per = fee_per.set_index('Date')
    hash_rate = pd.read_csv(hash_r, header = 0)
    hash_rate['Date'] = pd.to_datetime(hash_rate['Date'])
    hash_rate = hash_rate.set_index('Date')
    block_trans = pd.read_csv(block_t, header = 0)
    block_trans['Date'] = pd.to_datetime(block_trans['Date'])
    block_trans = block_trans.set_index('Date')
    
    # Join all series together on date
    chain_info = daily_trans.join(difficulty)
    chain_info = chain_info.join(fee_per)
    chain_info = chain_info.join(hash_rate)
    chain_info = chain_info.join(block_trans)
    
    return chain_info

def create_features (econ_data, chain_info_data):
    features = econ_data.join(chain_info_data)
    features = features.iloc[::-1]
    features = features.reset_index()
    return features

## Load Technical Analysis Features##
def rsi(df):
    # Returns info only as of 15th day in series as no price diff information available for first day!!
    days = 14
    close = df['Close']
    diff = close.diff()
    up, down = diff.copy(), diff.copy()
    up[up < 0] = 0
    down[down > 0] = 0
    up_mean = pd.rolling_mean(up, days)
    down_mean = pd.rolling_mean(down.abs(), days)
    RS = up_mean / down_mean
    RSI = 100.0 - (100.0 / (1.0 + RS))
    return RSI

def williams_r(df):
    # closing price in relation to past 13 days and closing day itself
    time_length = 13
    close = df['Close']
    high = df['High']
    low = df['Low']
    lst = []
    
    for i in range(len(close)):
        hh = 0
        ll = close.max()
        c = close.iloc[i]
        hh = high.iloc[(i - time_length): i+1].max()
        ll = low.iloc[(i - time_length): i+1].min()
        r = ((hh - c) / (hh - ll)) * (-100)
        lst.append(r)
    return pd.Series(lst) 


def ema_5_10(df):
    close = df['Close']
    ewma5 = pd.stats.moments.ewma(close, 5)
    ewma10= pd.stats.moments.ewma(close, 10)
    return ewma5, ewma10

def stochastic_k(df):
    # closing price in relation to past 13 days and closing day itself
    time_length = 13
    close = df['Close']
    high = df['High']
    low = df['Low']
    lst = []
    
    for i in range(len(close)):
        hh = 0
        ll = close.max()
        c = close.iloc[i]
        hh = high.iloc[(i - time_length): i+1].max()
        ll = low.iloc[(i - time_length): i+1].min()
        k = ((c - ll) / (hh - ll)) * (100)
        lst.append(k)
    return pd.Series(lst) 

def momentum(df):
    days = 4
    close = df['Close']
    lst = []
    for i in range(len(close)):
        if i < days:
            lst.append(0)
        else:
            m = close.iloc[i] - close.iloc[i - days]
            lst.append(m)
    return pd.Series(lst)

def stochastic_d(stochastic_k):
    days = 3
    lst = []
    for i in range(len(stochastic_k)):
        K = stochastic_k.iloc[(i-days)+1:i+1].sum()
        lst.append((K/3))
    return pd.Series(lst)

def slow_d(stochastic_d):
    #simply stochastic_d slowed by finding 3 day average
    days = 3
    lst = []
    for i in range(len(stochastic_d)):
        K = stochastic_d.iloc[(i-days)+1:i+1].sum()
        lst.append((K/3))
    return pd.Series(lst)

def rate_of_change(df):
    days = 12
    close = df['Close']
    lst = []
    for i in range(len(close)):
        roc = ((close.iloc[i] - close.iloc[i - 12]) / (close.iloc[i - 12])) *100
        lst.append(roc)
    return pd.Series(lst)

def ad_oscillator(df):
    close = df['Close']
    high = df['High']
    low = df['Low']
    lst = []
    for i in range(len(close)):
        ad = (high.iloc[i] - close.iloc[i-1]) / (high.iloc[i] - low.iloc[i])
        lst.append(ad)
    return pd.Series(lst) 

def macd(df):
    close = df['Close']
    ewma12 = pd.stats.moments.ewma(close, 12)
    ewma24 = pd.stats.moments.ewma(close, 24)
    return (ewma12 - ewma24) 

def y_labels(df):
    df['price'] = ''
    for i in range(1,len(df)-1):
        df.set_value(i, 'price', df['Close'].iloc[i+1])
    
    df['diff'] = df['Close'].diff()
    df['rise_fall'] = ''
    for i in range(1,len(df)-1):
        if df['diff'].iloc[i+1] >= 0:
            df.set_value(i, 'rise_fall', 1)
        elif df['diff'].iloc[i+1] < 0:
            df.set_value(i, 'rise_fall', 0)
        
    df['log_price'] = ''
    for i in range(1,len(df)-1):
        df.set_value(i, 'log_price', np.log10(df['price'].iloc[i]))
    
    return df


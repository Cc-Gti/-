# -*- coding: utf-8 -*-
"""
Created on Fri Dec  9 10:08:03 2022

@author: Lenovo
"""

import numpy as np
import pandas as pd
import time
from scipy import interpolate

#######替换NAN#############
def fillNaN_with_unifrand(df):
    """info:替换nan,用当前数组的均值和方差的随机数填充nan
    df:类型为Series、Dataframe,为单列"""
    a = df.values
    m = np.isnan(a) # mask of NaNs
    mu, sigma = df.mean(), df.std()
    a[m] = np.random.normal(mu, sigma, size=m.sum())
    return df 
#######替换0###############
def fillzero_with_unifrand(df):
    """info:替换nan,用当前数组的均值和方差的随机数填充nan
    df:类型为Series、Dataframe,为单列"""
    a = df.values
    m = np.where(a==0)[0] #返还的是tuple,所以添加[0]
    mu, sigma = df.mean(), df.std()
    try:
        a[m] = np.random.normal(mu, sigma, size=len(m))
    except ValueError:pass
    try:
        a[m] = np.random.normal(mu, sigma, size=len(m)).reshape(-1,1)
    except ValueError:pass    
    return df 
#######替换固定值val########
def fillval_with_unifrand(df,val):
    """info:替换固定值val
    df:类型为Series、Dataframe,为单列"""    
    a = df.values
    m = np.where(a==val)[0] #返还的是tuple,所以添加[0]
    mu, sigma = df.mean(), df.std()
    try:
        a[m] = np.random.normal(mu, sigma, size=len(m)).reshape(-1,1)
    except ValueError:pass
    try:    
        a[m] = np.random.normal(mu, sigma, size=len(m))
    except ValueError:pass    
    return df 

#######替换n倍标准差之外的所有数据######
def fill_nsigma_with_unifrand(df,n):
    """info:替换n倍标准差之外的所有数据
    df:类型为Series、Dataframe,为单列"""
    a = df.values
    mu, sigma = df.mean(), df.std()
    m = np.where((a>mu+n*sigma)&(a<mu-n*sigma))[0] #返还的是tuple,所以添加[0]
    try:    
        a[m] = np.random.normal(mu, sigma, size=len(m)).reshape(-1,1)
    except ValueError:pass
    try:    
        a[m] = np.random.normal(mu, sigma, size=len(m))
    except ValueError:pass        
    return df 




def fill_mostnum_with_unifrand(df):####某一列替换重复次数最多的值
     """info:替换重复次数最多的值，用于处理数据中心的水平直线线
       df:类型为Series、Dataframe,为单列"""
     mu, sigma = df.mean(), df.std()
        
     max_count=df.value_counts().values[0]#重复最多的数据的次数
     
     value=df.value_counts().index[0]#重复最多的数据
     
     repeat_index = df[df == value].index.tolist()
     
     print('重复最多的数：%s'%value)
     print('重复次数：%s'%max_count)
     print( df.loc[repeat_index].values.shape)
     
     a=df.values
     
     try:
         a[repeat_index]=np.random.normal(mu, sigma/3, size=len(repeat_index))
     except ValueError:pass
     try:     
         a[repeat_index]=np.random.normal(mu, sigma/3, size=len(repeat_index)).reshape(-1,1)
     except ValueError:pass     
     return df    
 
def fillnan_with_interplot(df,method='linear'):
    pass


def df_to_numd(df):    
    """
    目的：用于替换异常值
    """
    dict_ori={'0':0.0,'1':1.0,'<1':0.0,'<15':14.0,'<0.03':0.02,'＜0.03':0.02,'>180':181.0,'>200':201.0,'>8.0':9.0,'<0.01':0.009,\
              '<-50':-51.0,'60{-a)-}':61.0,'>16':17,'>11':12,'＞11':12,'--':np.nan,' ':np.nan,'1a':1.0,'1b':2.0,'痕迹':0.0,\
              '2a':3.0,'2b':4.0,'2c':5.0,'2d':6.0,'2e':7.0,'3a':8.0,'3b':9.0,'4a':10.0,'4b':11.0,'4c':12.0,\
              '>6000':6001,'>18000':18001,'>4000':4001,'>12000':12001,'<0.03（痕迹）':0.02,'>12':13\
                  ,'>16{-a)-}':17,'11{-a)-}':12,'--{-a)-}':17,'9{-a)-}':10,'10{-a)-}':11}
    for key,value in dict_ori.items():
        df=df.replace(key,value)
    return df 
     
      
def pdinterpolatedata(df,arg):
    """
    df:dataframe
    arg：检测项目，如黏度、水分、水活性等
    
    原理及作用：
    通过插值法，解决数据中存在缺失值的现象。
     
    
    return ：处理后的df
    info：
    当数据长度>4时，采用二次差值，否则采用线性插值

    """    
    if '时间' in df.columns:
        
        df['时间'] = pd.to_datetime(df['时间'])
        df = df.sort_values(by = '时间')
    if 'Time' in df.columns:
        df['Time'] = pd.to_datetime(df['Time'])
        df = df.sort_values(by = 'Time')         
    _df=df
    
    _df=df_to_numd(_df)
    
    _df[arg]=pd.to_numeric( _df[arg],errors='coerce')
    
    _df=_df.dropna(axis=0,how='any')
    
    #print(_df)
    if '时间' in df.columns:
    
        helper = pd.DataFrame({'时间': pd.date_range(_df['时间'].min(), _df['时间'].max())})
        
    
        dfnew = pd.merge(_df, helper, on='时间', how='outer').sort_values('时间')
    if 'Time' in df.columns:        
    
        helper = pd.DataFrame({'Time': pd.date_range(_df['Time'].min(), _df['Time'].max())})
        
    
        dfnew = pd.merge(_df, helper, on='Time', how='outer').sort_values('Time')    #print( dfnew)
    
    dfnew=dfnew.reset_index()
    
    x=dfnew.dropna(axis=0,how='any').index
    
    y=dfnew.dropna(axis=0,how='any')[arg].values    
    
       
    xnew=dfnew.index
    
    #print('xnew:',xnew)
    
    if len(y)>4:
    
        f=interpolate.interp1d(x, y,kind='quadratic',fill_value="extrapolate")
        
    else: 
        
        f=interpolate.interp1d(x, y,kind='linear',fill_value="extrapolate")   
    
    ynew=f(xnew)    
    
 
    dfnew[arg] = ynew

    if '时间' in df.columns:      
         dfnew=dfnew.set_index('时间')
    if 'Time' in df.columns:          
         dfnew=dfnew.set_index('Time')    
    
    #print('predeal done')
    
    return dfnew    
    
    
    
if __name__ == '__main__':
    pass
    
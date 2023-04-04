# -*- coding: utf-8 -*-
"""
Created on Mon Nov  7 16:31:39 2022

@author: Lenovo
"""

import pandas as pd
import matplotlib.pyplot as plt
from adtk.detector import LevelShiftAD,PersistAD,VolatilityShiftAD
import os 

from log import log
from ZX_Data_deal import pdinterpolatedata






def unnomal_point_check1(ts,window=30,c=1.0,side='positive'):
    """
    ts:series
    indow：参考窗长度，可为int, str
    c:分位距倍数，用于确定上下限范围
    side：[positive','negative','both'],检测范围，为'positive'时检测突增，为'negative'时检测突降，为'both'时突增突降都检测
    
    原理：
    用滑动窗口遍历历史数据，将窗口后的一位数据与参考窗中的统计量做差，得到一个新的时间序列s1;
    计算s1的(Q1-c*IQR, Q3+c*IQR) 作为正常范围；
    若当前值与它参考窗中的统计量之差，不在2中的正常范围内，视为异常。
     
    info：
    window：越大，模型越不敏感，不容易被突刺干扰a
    c：越大，对于波动大的数据，正常范围放大较大，对于波动较小的数据，正常范围放大较小
    总结：先计算一条新的时间序列，再用箱线图作异常检测
    """
    try:
        persist_ad = PersistAD(c=c, side=side)
        anomalies = persist_ad.fit_detect(ts)
        outliers_indices=anomalies[anomalies==1]
        result=ts[outliers_indices.index]  
        ##以下是可视化,不需要就注释掉           
        plt.figure(figsize=(10,4))
        plt.plot(ts, label = 'ts_origin') #原序列图
        plt.plot(result, 'ro', label='anomaly') #给异常值描红点
        plt.xticks(rotation = 45)
        plt.legend()
        plt.title(ts.name)
        plt.show()
        print('unnomal_point_check1 Done')
    except Exception as e:
        log(e)
        result = e
        print('unnomal_point_check1 Fail')        
    return result

def unnomal_point_check2(ts,window=30,c=6.0,side='positive'):
    """
    ts:series
    indow：参考窗长度，可为int, str
    c:分位距倍数，用于确定上下限范围
    side：[positive','negative','both'],检测范围，为'positive'时检测突增，为'negative'时检测突降，为'both'时突增突降都检测
    
    原理：
         用于突变点检测，跟踪两个相邻滑动时间窗口的中位数值之间的差异来检测值水平的偏移，它对瞬时峰值不敏感，如果经常发生嘈杂的异常值，它可能是一个不错的选择
     
    info：
    window：支持(10,5)，表示使用两个相邻的滑动窗，左侧的窗中的中位值表示参考值，右侧窗中的中位值表示当前值
    c：越大，对于波动大的数据，正常范围放大较大，对于波动较小的数据，正常范围放大较小，默认6.0
    side：检测范围，为'positive'时检测突增，为'negative'时检测突降，为'both'时突增突降都检测
    """
    try:    
        level_shift_ad = LevelShiftAD(c=c, side=side, window=window)
        anomalies = level_shift_ad.fit_detect(ts)
        #anomalies = persist_ad.fit_detect(ts)
        outliers_indices=anomalies[anomalies==1]
        result=ts[outliers_indices.index]  
        ##以下是可视化,不需要就注释掉        
        plt.figure(figsize=(10,4))
        plt.plot(ts, label = 'ts_origin') #原序列图
        plt.plot(result, 'ro', label='anomaly') #给异常值描红点
        plt.xticks(rotation = 45)
        plt.legend()
        plt.title(ts.name)
        plt.show()
        print('unnomal_point_check2 Done')
    except Exception as e:
        log(e)
        result = e
        print('unnomal_point_check2 Fail')              
    return result


def unnomal_point_check3(ts,window=30,c=6.0,side='positive'):
    """
    ts:series
    indow：参考窗长度，可为int, str
    c:分位距倍数，用于确定上下限范围
    side：[positive','negative','both'],检测范围，为'positive'时检测突增，为'negative'时检测突降，为'both'时突增突降都检测
    
    原理：
         用于突变点检测，通过跟踪两个相邻滑动时间窗口下标准偏差之间的差异来检测波动率水平的变化判定异常
     
    info：
    window：支持(10,5)，表示使用两个相邻的滑动窗，左侧的窗中的中位值表示参考值，右侧窗中的中位值表示当前值
    c：越大，对于波动大的数据，正常范围放大较大，对于波动较小的数据，正常范围放大较小，默认6.0
    side：检测范围，为'positive'时检测突增，为'negative'时检测突降，为'both'时突增突降都检测
    """
    try:     
        volatility_shift_ad = VolatilityShiftAD(c=c, side=side, window=window)
        anomalies = volatility_shift_ad.fit_detect(ts)
        outliers_indices=anomalies[anomalies==1]
        result=ts[outliers_indices.index] 
        ##以下是可视化,不需要就注释掉
        plt.figure(figsize=(10,4))
        plt.plot(ts, label = 'ts_origin') #原序列图
        plt.plot(result, 'ro', label='anomaly') #给异常值描红点
        plt.xticks(rotation = 45)
        plt.legend()
        plt.title(ts.name)
        plt.show()
        print('unnomal_point_check3 Done')
    except Exception as e:
        log(e)
        result = e
        print('unnomal_point_check3 Fail')          
    return result








       
if __name__ == '__main__':
    
    plt.rcParams['font.sans-serif']=['SimHei']
    plt.rcParams['axes.unicode_minus'] = False

    project_path = os.path.split(os.path.realpath(__file__))[0]
    df_data=pd.read_csv(project_path+'\data\DY401.csv',encoding='gbk')  

    df=pdinterpolatedata(df_data,'黏度')
    unnomal_point_check3(df['黏度'],window=50,c=6.0,side='positive')
    unnomal_point_check2(df['黏度'],window=60,c=2.0,side='positive')    
    unnomal_point_check1(df['黏度'],window=60,c=2.0,side='positive')        
    
    
    
    
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 18 16:36:03 2020

@author: Cuice
"""
#所有函数中的arg为列名称，比如水分，黏度，油温，保存图片可调用用plt.savefig()函数
import time
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from sklearn import linear_model
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima_model import ARMA
from statsmodels.stats.diagnostic import acorr_ljungbox
import scipy.stats as st
sns.set(style="darkgrid")
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus'] = False



def var_cal(arg,df):#某一列均值、方差、标准差var_cal('黏度',df_data)
    comment="当前%s均值为%s,标准差为%s,中位数为%s"%(arg,df[arg].mean(),df[arg].std(),df[arg].median())
    return comment

def box_plt(arg,df):#此函数画盒图 box_plt('黏度',df_data)
    df[[arg]].boxplot()
    plt.text(0.5,df[[arg]].values.mean(), r'$\mu$={:.2f}, $\sigma$={:.2f}'.format(df[[arg]].values.mean(), df[[arg]].values.std()) ) #text的位置未确定，此行可以删除
    plt.show()

def plt_date_tendency(date1,date2,arg,df):#画出date1~date2之间的数据趋势,并给出相应的注意区间 plt_date_tendency('2019-09-05 12:05:00','2019-09-05 23:35:00','黏度',df_data)
    
    if 'time' in df.columns:
        df['time'] = pd.to_datetime(df['time'])  
        df=df.set_index('time')
    if '时间' in df.columns: 
        df['时间'] = pd.to_datetime(df['时间'])        
        df=df.set_index('时间')
    df=df.sort_index()   
    df[date1:date2][arg].plot(figsize=(12,6))   #
    plt.xticks(rotation=70)
    plt.fill_between( np.array(df.index),df[arg].mean()-df[arg].std(), df[arg].mean()+df[arg].std(), facecolor='blue', alpha=0.3)
    plt.fill_between( np.array(df.index),df[arg].mean()+df[arg].std(), df[arg].mean()+2*df[arg].std(), facecolor='orange', alpha=0.4)
    plt.fill_between( np.array(df.index),df[arg].mean()-df[arg].std(), df[arg].mean()-2*df[arg].std(), facecolor='orange', alpha=0.4)
    #plt.autofmt_xdate()
    plt.show()
    
def draw_trend(date1,date2,arg,df,size=100,method='mean'):#移动平均及加权移动平均,；例如size=3#draw_trend('黏度',3,df_data)
    """
    date1：起始时间
    date2:截止时间
    arg:黏度、油温
    df：数据
    size:窗口大小
    method:['mean','median'],默认mean均值
    info:会修改原始数据df"""
    if 'time' in df.columns:
        df['time'] = pd.to_datetime(df['time'])  
        df=df.set_index('time')
    if '时间' in df.columns: 
        df['时间'] = pd.to_datetime(df['时间'])        
        df=df.set_index('时间')
    df=df.sort_index()
    timeSeries=df[date1:date2][arg]#增加了时间
#    f = plt.figure(facecolor='white')
    if method=='mean':
        rol_mean = timeSeries.rolling(window=size).mean()# 对size个数据进行移动平均
        rol_weighted_mean = pd.DataFrame.ewm(timeSeries, span=size).mean()#加权移动平均

    if method=='median':
        rol_mean = timeSeries.rolling(window=size).median()# 对size个数据进行移动平均

        rol_weighted_mean = pd.DataFrame.ewm(timeSeries, span=size).mean()#加权移动平均  
      
    rol_mean.plot(color='red', label='Rolling Mean',figsize=(12,6))
    rol_weighted_mean.plot(color='black', label='Weighted Rolling Mean')
    timeSeries.plot(color='blue', label='Original')
    plt.legend(loc='best')
    plt.xlabel('time')
    plt.xticks(rotation=70)
    plt.title(arg)
    plt.show()
    
def draw_trend1(date1,date2,arg,df,size=100,method='mean'):#移动平均及加权移动平均,；例如size=3#draw_trend('黏度',3,df_data)
    """
    date1：起始时间
    date2:截止时间
    arg:黏度、油温
    df：数据
    size:窗口大小
    method:['mean','median'],默认mean均值
    info:会修改原始数据df"""
    
    if 'time' in df.columns:
        df['time'] = pd.to_datetime(df['time'])  
        df=df.set_index('time')
    if '时间' in df.columns: 
        df['时间'] = pd.to_datetime(df['时间'])        
        df=df.set_index('时间')
    df=df.sort_index()
    timeSeries=df[date1:date2][arg]
    if method=='mean':
        rol_mean = timeSeries.rolling(window=size).mean()# 对size个数据进行移动平均

    if method=='median':
        rol_mean = timeSeries.rolling(window=size).median()# 对size个数据进行移动平均

    rol_mean.plot(color='red', label='Rolling Mean',figsize=(12,6))

    timeSeries.plot(color='blue', label='Original')
    plt.legend(loc='best')
    plt.xlabel('time')

    plt.xticks(rotation=70)
    plt.title(arg)
    plt.show()    
def density_plt(arg,df):#此函数是画概率分布 density_plt('黏度',df_data)    
     for i in range(len(df)):
         sns.distplot(df[[arg]].dropna(), kde=True)
         #plt.text(df[[arg]].values.mean(), 0.062, r'$\mu$={:.2f}, $\sigma$={:.2f}'.format(df[[arg]].values.mean(), df[[arg]].values.std()) ) #text的位置未确定，此行可以删除
         plt.xlabel(arg)
         plt.axvline(x=df[arg][i],ls="-",c="green")#添加垂直直线
         plt.show()

def density_plt2(arg,df):#此函数是画概率分布 density_plt('黏度',df_data)    
    sns.distplot(df[[arg]].dropna(), kde=True)
    plt.xlabel(arg)
    plt.show() 
def corelation_plt(df,**kwargs):#关联性图,corelation_plt(df_data,myarg1='油温',myarg2='水活性',myarg3='黏度')
    # df=df.set_index('time')
     start = time.time()
     df2=pd.Series(data=np.random.randint(1,2,size=len(df)))
     try:
         for key in kwargs:
           if kwargs[key]=='':
              pass
           else:
                df3=df[kwargs[key]]
                df2= pd.concat([df2, df3], axis=1)
     except KeyError:pass   
     df3=df2.drop([0], axis=1)
     sns.heatmap(df3.corr(),annot=True)
     sns.pairplot(df3.dropna(),kind='reg')#hue=arg 
     #sns.pairplot(df3.dropna())#hue=arg 
     plt.savefig('D:\Desktop\ssss.png')
     plt.show()
    
     end = time.time()
     running_time = end-start
     print('time cost : %.5f sec' %running_time)
     
def trend_cal(date1,date2,arg,df):#需要在上个函数测试不平稳的情况下，才能使用该函数trend_cal('2019-09-05 12:05:00','2019-09-05 23:35:00','黏度',df_data)
    if 'time' in df.columns:
        df['time'] = pd.to_datetime(df['time'])  
        df=df.set_index('time')
    if '时间' in df.columns: 
        df['时间'] = pd.to_datetime(df['时间'])        
        df=df.set_index('时间')
    df=df.sort_index()
    x=np.linspace(1,len(df[date1:date2]),num=len(df[date1:date2])).reshape(-1,1)
    y=df[date1:date2][arg].values.reshape(-1,1)
    #df[date1:date2][arg].plot()
    model = linear_model.LinearRegression()
    model.fit(x, y)
    y2=model.predict(x)
    plt.plot(x, y2, 'g-')
    plt.show()
    if model.coef_>0:
       trend_comment='最小二乘法拟合趋势项，得出k=%s,表明%s有增长趋势'%(model.coef_[0],arg) 
    if model.coef_<0:
       trend_comment='最小二乘法拟合趋势项，得出k=%s,表明%s有下降趋势'%(model.coef_[0],arg) 

    return trend_comment, model.coef_[0]

def jiexi(date1,date2,arg,df):#上个函数得出的结论不平稳才解析，把时间序列分层趋势项、噪音项目及周期项目 jiexi('2019-09-05 12:05:00','2019-09-05 23:35:00','黏度',df_data)
    if 'time' in df.columns:
        df['time'] = pd.to_datetime(df['time'])  
        df=df.set_index('time')
    if '时间' in df.columns: 
        df['时间'] = pd.to_datetime(df['时间'])        
        df=df.set_index('时间')
    df=df.sort_index()
    decomposition = seasonal_decompose(df[date1:date2][arg],model="additive",period = 2)#Y30？????为什么，怎么定#, model="multiplicative",period = 2
    decomposition.plot()
    trend = decomposition.trend
    trend.plot()
    plt.title('trend')
    plt.savefig('D:/Desktop/web-python/trend.png',bbox_inches = 'tight')
    plt.show()

    seasonal = decomposition.seasonal
    seasonal.plot()
    plt.title('seasonal')
    plt.savefig('D:/Desktop/web-python/seasonal.png',bbox_inches = 'tight')
    plt.show()
     

    residual = decomposition.resid
    p_1,p_2=acorr_ljungbox(residual ,lags = [6, 12],boxpierce=False)      
    residual.plot()
    plt.title('residual')
    plt.savefig('D:/Desktop/web-python/residual.png',bbox_inches = 'tight')
    plt.show()
    if p_2[0]<0.05:
       res_commentr='噪音项为非白噪声序列，说明系统可能存在外来干扰。' 
    else:
       res_commentr='噪音项为白噪声序列，可能源于系统误差。' 
    return res_commentr
def jiexi2(date1,date2,arg,df):#上个函数得出的结论不平稳才解析，把时间序列分层趋势项、噪音项目及周期项目 jiexi('2019-09-05 12:05:00','2019-09-05 23:35:00','黏度',df_data)
    """对在线数据进行分析，得出文字型结论
        date1:开始时间，datetime形式
        date2:开始时间，datetime形式
        df:dataframe
        arg:监测参数如黏度、水分
        return:返还图像
    """         
    if 'time' in df.columns:
        df['time'] = pd.to_datetime(df['time'])  
        df=df.set_index('time')
    if '时间' in df.columns: 
        df['时间'] = pd.to_datetime(df['时间'])        
        df=df.set_index('时间')
    df=df.sort_index()
    
    decomposition = seasonal_decompose(df[date1:date2][[arg]],model="additive",)#period = 2  Y30？????为什么，怎么定#, model="multiplicative",period = 2
    trend = decomposition.trend
    trend.plot(figsize=(12,6))
    plt.title('trend')
    plt.savefig('D:/Desktop/web-python/trend.png',bbox_inches = 'tight')
    plt.show()

    seasonal = decomposition.seasonal
    seasonal.plot(figsize=(12,6))
    plt.title('seasonal')
    plt.savefig('D:/Desktop/web-python/seasonal.png',bbox_inches = 'tight')
    plt.show()
     

    residual = decomposition.resid
    p_1,p_2=acorr_ljungbox(residual ,lags = [6, 12],boxpierce=False)      
    residual.plot(figsize=(12,6))
    plt.title('residual')
    #plt.savefig('D:/Desktop/web-python/residual.png',bbox_inches = 'tight')
    plt.show()
    if p_2[0]<0.05:
       res_commentr='噪音项为非白噪声序列，说明系统可能存在外来干扰。' 
    else:
       res_commentr='噪音项为白噪声序列，可能源于系统误差。' 
    return res_commentr
def testStationarity(date1,date2,arg,df):#增加时间?testStationarity('2019-09-05 12:05:00','2019-09-05 23:35:00','黏度',df_data)
    """对在线数据进行平稳性分析以及趋势分析，得出文字型结论
        date1:开始时间，datetime形式
        date2:开始时间，datetime形式
        df:dataframe
        arg:监测参数如黏度、水分
        return:返还str以及一个ADF检验的列表
        
    """        

    if 'time' in df.columns:
        df['time'] = pd.to_datetime(df['time'])  
        df=df.set_index('time')
    if '时间' in df.columns: 
        df['时间'] = pd.to_datetime(df['时间'])        
        df=df.set_index('时间') 
    df=df.sort_index()
    ts=df[date1:date2][arg]
    dftest = adfuller(ts)
    print(1)
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])  
    for key,value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
    print(dfoutput)
    if (dfoutput['p-value']<0.05 and dfoutput['Test Statistic']<dfoutput['Critical Value (1%)']) or \
        np.isnan(dfoutput['p-value']) or np.isnan(dfoutput['Test Statistic']):
        dfcomment1='%s序列平稳，ADF检测值小于1%%的显著水平，p<0.05' %arg
    elif dfoutput['p-value']<0.05  and  dfoutput['Critical Value (1%)']<dfoutput['Test Statistic']<dfoutput['Critical Value (5%)']:
        dfcomment1='%s序列平稳，ADF检测值小于5%%的显著水平，p<0.05' %arg
    elif dfoutput['p-value']<0.05  and  dfoutput['Critical Value (5%)']<dfoutput['Test Statistic']<dfoutput['Critical Value (10%)']:
        dfcomment1='%s序列较为平稳，ADF检测值小于10%%的显著水平，p<0.05'%arg 
    elif dfoutput['p-value']>0.05  or  dfoutput['Test Statistic']>dfoutput['Critical Value (10%)']: 
        dfcomment1='%s序列不平稳，ADF检验值大于任意显著水平，p>0.05'%arg    
        decomposition=seasonal_decompose(df[date1:date2][arg],model="additive", period = 2)#Y30？????为什么，怎么定#, model="multiplicative",period = 2          
        trend = decomposition.trend
        x=np.linspace(1,len(trend.dropna()),num=len(trend.dropna())).reshape(-1,1)
        y=trend.dropna().values.reshape(-1,1)
        model = linear_model.LinearRegression()
        model.fit(x, y)
        if model.coef_>0:
           dfcomment1=dfcomment1+',序列整体有上升趋势。'
        if model.coef_<0:
           dfcomment1=dfcomment1+',序列整体有下降趋势。'
     
    return dfoutput,dfcomment1

def corr_comment(date1,date2,arg,df):#corr_comment('2019-09-05 12:05:00','2019-09-05 23:35:00','黏度',df_data)
    """对在线数据进行分析，得出文字型结论
        date1:开始时间，datetime形式
        date2:开始时间，datetime形式
        df:dataframe
        arg:监测参数如黏度、水分
        return:返还str
        
    """    

    if 'time' in df.columns:
        df['time'] = pd.to_datetime(df['time'])  
        df=df.set_index('time')
    if '时间' in df.columns: 
        df['时间'] = pd.to_datetime(df['时间'])   
    #a,b=testStationarity(date1,date2,arg,df)
    #df=df.set_index('time')
    df=df.sort_index()
    corr=df[date1:date2].corr(method='pearson').round(2) 
    corr=corr[abs(corr)>0.5].to_dict()
    corr.pop(arg,'index')
    list1=[]
    if corr!={}:    
        for v,k in corr.items():
            list1.append('{v}:r={k}'.format(v = v, k = k))   
            str1=('、'.join(list1))   
            corrcomment='{}~{},与{}相关的属性及相关系数是{}'.format(date1,date2,arg,str1)
    else: corrcomment='%s与其它属性关联性较低，关联系数r的绝对值均小于0.5'%arg          
    return corrcomment

def corr_comment1(date1,date2,df):#corr_comment('2019-09-05 12:05:00','2019-09-05 23:35:00',df_data)
    """对在线数据进行分析，得出文字型结论
        date1:开始时间，datetime形式
        date2:开始时间，datetime形式
        df:dataframe
        
    """
    df=df.set_index('time')
    df=df.sort_index()
    corr=df[date1:date2].corr(method='pearson').round(2) 
    for i in ['油温','NAS等级','含水量','黏度']:
        corr2=corr[i][abs(corr)>0.5].to_dict()
        corr2.pop(i) 
        list2=[]
        if corr2!=[]:
          for v,k in corr2.items():  
              list2.append('{z}与{v}:r={k}'.format(z=i,v = v, k = k)) 
              for j in list2:
                  pass
def viscosty(date1,date2,df):#viscosty('2019-09-05 12:05:00','2019-09-05 23:35:00',df_data)
    """"黏度评价函数"""
    a,b=testStationarity(date1,date2,'黏度',df)
    viscommnet='黏度正常'
    if '不平稳' in b:
        viscommnet='黏度序列不平稳，表明导致油膜强度不稳，局部油膜被击穿的概率增加；建议关注黏度变化情况。'
    return viscommnet        
def water(date1,date2,df):#water('2019-09-05 12:05:00','2019-09-05 23:35:00',df_data)
    """"黏度评价函数"""    
    a,b=testStationarity(date1,date2,'含水量',df)
    watercommnet='水分含量正常'
    if '上升' in b:
        watercommnet='含水量升高，可能由于密封不严导致外部空气水分浸入，或存在冷却水泄漏，使得含水量升高；建议关注水分变化情况，检查邮箱底部是否存在游离水或乳化液，必要时采取脱水处理。'
    elif  '下降' in b: 
        watercommnet='含水量下降，可能由于油液分水性导致油中部分水分分离；建议检查邮箱底部是否存在游离水或乳化液。'
    return watercommnet    
def NAS(date1,date2,df):#NAS('2019-09-05 12:05:00','2019-09-05 23:35:00',df_data)
    """"黏度评价函数"""    
    a,b=testStationarity(date1,date2,'NAS等级',df)
    NAScommnet='污染度正常'
    if '上升' in b:
        NAScommnet='污染度等级升高，可能由于密封不严导致外部粉尘进入,建议加强过滤净化处理。'
    return NAScommnet    
def temp(date1,date2,df):#temp('2019-09-05 12:05:00','2019-09-05 23:35:00',df_data)
    """"黏度评价函数"""    
    a,b=testStationarity(date1,date2,'油温',df)
    tempcommnet='油温正常'
    if '上升' in b:
        tempcommnet='油温升高，表明系统可能存在冷却水流量不足或摩擦副异常磨损；建议检查设备运行情况。'
    return tempcommnet
def Fe(date1,date2,df):#Fe('2019-09-05 12:05:00','2019-09-05 23:35:00',df_data)
    """"磨损预测"""    
    a1,b1=testStationarity(date1,date2,'Fe70~100',df)
    a2,b2=testStationarity(date1,date2,'Fe100~150',df)
    a3,b3=testStationarity(date1,date2,'Fe>150',df)
    a4,b4=testStationarity(date1,date2,'nFe200~300',df)
    a5,b5=testStationarity(date1,date2,'nFe300~400',df)
    a6,b6=testStationarity(date1,date2,'nFe>400',df)
    tempcommnet='磨损含量正常'
    if '上升' in b1 or '上升' in b2 or '上升' in b3 or '上升' in b4 or '上升' in b5 or '上升' in b6:
        tempcommnet='油中Fe颗粒含量升高，表明系统可能存在异常磨损；建议检查设备运行情况。'
    return tempcommnet

def trend_warning_strategy(df):
    """用于趋势报警,检验df[arg]是否有连续增长趋势，若是，返还true,否，返还false
        df:series"""
    #diff_sum=df[arg].diff().dropna().sum()
    diff_abs=df.diff().dropna()
    if diff_abs.all()>0:
       tag=True
    else:tag=False
    return tag




def cal_cdf(data):
    """data:pandas.core.series.Series,or Dataframe,来自pandas模块
       返还累积分布95%、98%的阈值"""
    bins_val=10   
    while bins_val:
        hist, bin_edges = np.histogram(data,bins=bins_val)
    
        cdf = np.cumsum(hist/sum(hist))
    
    #0.95分位值
        index95_2=np.where(cdf> 0.95)[0][0]
        if index95_2==0:
           bins_val+=5
        else:break   
    print(bins_val)
    index95_1=index95_2-1
    k1=(cdf[index95_2]-cdf[index95_1])/(bin_edges[index95_2+1]-bin_edges[index95_1+1])

    val_95=(0.95-cdf[index95_1])/k1+bin_edges[index95_1+1]
    #0.98分位值
    
    index98_2=np.where(cdf> 0.98)[0][0]

    index98_1=index98_2-1
    k2=(cdf[index98_2]-cdf[index98_1])/(bin_edges[index98_2+1]-bin_edges[index98_1+1])

    val_98=(0.98-cdf[index98_1])/k2+bin_edges[index98_1+1]

    print('累积分布百分之95阈值%s'%round(val_95,2))    

    print('累积分布百分之98阈值%s'%round(val_98,2))
    
    return round(val_95,2),round(val_98,2)


def cdl_plt(df):
        

    for i in df.columns:
        hist, bin_edges = np.histogram(df[i])
        fig=plt.figure(figsize=(11,8))
        width = (bin_edges[1] - bin_edges[0]) * 0.8
        #plt.bar(bin_edges[1:], hist/max(hist), width=width, color='#5B9BD5')
        #df[i].plot.bar()
        #plt.hist(df[i], color='#5B9BD5')
            # 绘制直方图
        ax1=fig.add_subplot(111)
        plt.rcParams.update({'font.size': 16})#####设置所有字体
        ax1=df[i].plot(kind = 'hist', bins = 30, color = 'steelblue', edgecolor = 'black', density = True, label = '直方图')
        # 绘制核密度图
        ax1=df[i].plot(kind = 'kde', color = 'red', label = '核密度估计')
        plt.legend(loc='upper left')
       
        
        plt.xlabel(i)
        plt.xlim(0,9)
        ax2 = ax1.twinx()
        cdf = np.cumsum(hist/sum(hist))
        ax2=plt.plot(bin_edges[1:], cdf, '-*', color='#ED7D31',label='累计分布cdf')
        plt.ylabel('Percent')
        plt.legend(loc='upper right')
#       plt.savefig('D:\Desktop\cdf.png')
        plt.title(i)
        plt.show()

###用于黏度的打分####
def vis_score(val,mean,std):
    if val<mean:
        prob=st.norm.cdf(val,loc=mean,scale=std)/0.5
    else:prob=(1-st.norm.cdf(val,loc=mean,scale=std))/0.5
    return prob.round(2)*100
###用于其它项目的打分####    
def rest_item_score(val,mean,std):
    prob=1-st.norm.cdf(val,loc=mean,scale=std)
    return prob.round(2)*100

if __name__ == '__main__':
    pass
    # path_prefix=r'D:\Desktop' 
    # df_data=pd.read_excel(r'D:\Desktop\123.xlsx',sheet_name='Sheet1')#
    # df_data=df_data.apply(pd.to_numeric,errors='ignore')
    
    # df_data['time'] = pd.to_datetime(df_data['time'])
    # density_plt2('油温',df_data)     
    # var_cal('油温',df_data)
    # box_plt('油温',df_data)
    # jiexi('2020-04-14 00:01:00','2020-04-15 23:51:00','黏度',df_data)
    # draw_trend('2020-04-14 00:01:00','2020-04-15 23:51:00','Fe70~100',df_data,size=100,method='mean')
    # plt_date_tendency('2020-04-14 00:01:00','2020-04-15 23:51:00','含水量',df_data)
    # corelation_plt(df_data,myarg1='油温',myarg2='含水量',myarg3='黏度',myarg4='NAS等级')
    # for item in ['黏度','含水量','油温','NAS等级']: 
    #      print(testStationarity('2020-04-14 00:01:00','2020-04-15 23:51:00',item,df_data))    
    #      #print(corr_comment('2020-04-14 00:01:00','2020-04-15 23:51:00',item,df_data))    
    # print(water('2020-04-14 00:01:00','2020-04-15 23:51:00',df_data))
    # print(NAS('2020-04-14 00:01:00','2020-04-15 23:51:00',df_data))
    # print(temp('2020-04-14 00:01:00','2020-04-15 23:51:00',df_data))
    # print(viscosty('2020-04-14 00:01:00','2020-04-15 23:51:00',df_data))         
# =============================================================================

    
    
    
    
    
    
    
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 15 09:03:40 2021

@author: CC,用于在线预测
"""


import pandas as pd
import matplotlib.pyplot as plt
from autots import AutoTS
from zaixian import testStationarity
from log import log
import os
import json
import time
from matplotlib.gridspec import GridSpec

def arma_forcast(df_data,arg,modelname='model.csv'):##
    """
    df_data:dataframe
    arg:预测的参数，如水分、黏度等
    modelname：'model.csv'模型信息文件名
    info:用于训练数据,返还模型信息文件，存储在model下，同时进行预测和可视化，图片保存在根目录\png下，返还预测的结果 类型为dataframe
    
    """
    try:
        if  '时间' in df_data.columns:
            df_data['时间'] = pd.to_datetime(df_data['时间'])
        if  'time' in df_data.columns:
            df_data['time'] = pd.to_datetime(df_data['time'])
        t1=time.time()    
        model_list = ['Theta','ARIMA']#'ETS','ZeroesNaive','ETS','GLS','GLM' 'GLS','GLM','LastValueNaive','RollingRegression',,'ARDL'
        forecast_length=len(df_data)//4
        model = AutoTS( 
            forecast_length=forecast_length, 
            frequency='infer', 
            prediction_interval=0.90, 
            ensemble=None, 
            max_generations=4, 
            num_validations=2, 
            validation_method='backwards', 
            model_list=model_list, 
            transformer_list='all', 
            models_to_validate=0.2, 
            drop_most_recent=1, 
            n_jobs='auto', 
        )   
        if  '时间' in df_data.columns:
            model = model.fit(df_data, date_col='时间', value_col=arg, id_col=None)
        if  'time' in df_data.columns:
            model = model.fit(df_data, date_col='time', value_col=arg, id_col=None) 
        prediction = model.predict() 
        forecasts_df = prediction.forecast
        # validation_results = model.results("validation")
        # prediction.plot(model.df_wide_numeric,
        #         series=model.df_wide_numeric.columns[0],figsize=(10, 6))
        t2=time.time() 
        print('花费时间;%s'%(t2-t1))
        if  'time' in df_data.columns:
            df_data=df_data.set_index('time')
        if  '时间' in df_data.columns:
            df_data=df_data.set_index('时间')
        project_path = os.path.split(os.path.realpath(__file__))[0]
        model.export_template(project_path+'\\model\\%s'%modelname, models='best',
                                n=15, max_per_model_class=3)  
        forecasts_up, forecasts_low = prediction.upper_forecast, prediction.lower_forecast    
        temp_df=pd.DataFrame({'origin':df_data[arg],'forcast':forecasts_df[arg],\
                             'forecasts_low':forecasts_low[arg],'forecasts_up':forecasts_up[arg]})    
                
        ############ 以下是可视化代码##########           
        fig = plt.figure(figsize=(12,5),
                      constrained_layout=True,#类似于tight_layout，使得各子图之间的距离自动调整【类似excel中行宽根据内容自适应】
                     )    
        
        gs = GridSpec(1, 5, figure=fig)                 
        ax1 = fig.add_subplot(gs[0, 0:3])         
    
        for col in temp_df:
            temp_df[col].plot(label=col) 
            plt.legend()  
        plt.ylabel(arg)
        ax2 = fig.add_subplot(gs[0, 3:5]) 
        for col in temp_df:
            temp_df[col].plot(kind='hist',label=col) 
            plt.legend()
        plt.xlabel(arg)
        plt.savefig(project_path+'\\png\\arma_forcat.png',bbox_inches ='tight') 
        plt.show()    
        print('arma_forcast Done')
    except Exception as e:
        log(e)
        print('arma_forcast Fail')   
        temp_df=pd.DataFrame({'origin':df_data[arg]})
    return temp_df



def arma_forcast2(df_data,arg,modelname='model.csv'):##
    """
    df_data:dataframe
    arg:预测的参数，如水分、黏度等
    modelname：'model.csv'模型信息文件名
    info:用于训练数据,返还模型信息文件，存储在model下，同时进行预测和可视化，图片保存在根目录\png下，返还预测的结果 类型为dataframe
    
    """
    try:
        if  '时间' in df_data.columns:
            df_data['时间'] = pd.to_datetime(df_data['时间'])
        if  'time' in df_data.columns:
            df_data['time'] = pd.to_datetime(df_data['time'])
        t1=time.time()    
        model_list = ['LastValueNaive']#'fast'#'ETS','ZeroesNaive','ETS','GLS','GLM' 'GLS','GLM','LastValueNaive','RollingRegression'
        forecast_length=len(df_data)//4
        model = AutoTS( 
            forecast_length=forecast_length, 
            frequency='infer', 
            prediction_interval=0.90, 
            ensemble=None, 
            max_generations=4, 
            num_validations=2, 
            validation_method='similarity', 
            model_list=model_list, 
            transformer_list='all', 
            models_to_validate=0.2, 
            drop_most_recent=1, 
            n_jobs='auto', 
        )   
        if  '时间' in df_data.columns:
            model = model.fit(df_data, date_col='时间', value_col=arg, id_col=None)
        if  'time' in df_data.columns:
            model = model.fit(df_data, date_col='time', value_col=arg, id_col=None) 
        prediction = model.predict() 
        forecasts_df = prediction.forecast
        # validation_results = model.results("validation")
        # prediction.plot(model.df_wide_numeric,
        #         series=model.df_wide_numeric.columns[0],figsize=(10, 6))
        t2=time.time() 
        print('花费时间;%s'%(t2-t1))
        if  'time' in df_data.columns:
            df_data=df_data.set_index('time')
        if  '时间' in df_data.columns:
            df_data=df_data.set_index('时间')
        project_path = os.path.split(os.path.realpath(__file__))[0]
        model.export_template(project_path+'\\model\\%s'%modelname, models='best',
                                n=15, max_per_model_class=3)  
        forecasts_up, forecasts_low = prediction.upper_forecast, prediction.lower_forecast    
        temp_df=pd.DataFrame({'origin':df_data[arg],'forcast':forecasts_df[arg],\
                             'forecasts_low':forecasts_low[arg],'forecasts_up':forecasts_up[arg]})    
                
        ############ 以下是可视化代码##########           
        fig = plt.figure(figsize=(12,5),
                      constrained_layout=True,#类似于tight_layout，使得各子图之间的距离自动调整【类似excel中行宽根据内容自适应】
                     )    
        
        gs = GridSpec(1, 5, figure=fig)                 
        ax1 = fig.add_subplot(gs[0, 0:3])         
    
        for col in temp_df:
            temp_df[col].plot(label=col) 
            plt.legend()  
        plt.ylabel(arg)
        ax2 = fig.add_subplot(gs[0, 3:5]) 
        for col in temp_df:
            temp_df[col].plot(kind='hist',label=col) 
            plt.legend()
        plt.xlabel(arg)
        plt.savefig(project_path+'\\png\\arma_forcat.png',bbox_inches ='tight') 
        plt.show()    
        print('arma_forcast Done')
    except Exception as e:
        log(e)
        print('arma_forcast Fail')   
        temp_df=pd.DataFrame({'origin':df_data[arg]})
    return temp_df




def model_train(df_data,arg,modelname='model.csv'):##
    """
    df_data:dataframe
    arg:预测的参数，如水分、黏度等
    modelname：'model.csv'模型信息文件名
    info:用于训练数据,返还模型信息文件，存储在model下
    
    """
    try:
        if  '时间' in df_data.columns:
            df_data['时间'] = pd.to_datetime(df_data['时间'])
          #  df_data=df_data.set_index('时间')
            #df=df_data.resample('2H').mean()
        if  'time' in df_data.columns:
            df_data['time'] = pd.to_datetime(df_data['time'])
           #df_data=df_data.set_index('time')    
            #df=df_data.resample('2H').mean()#.fillna(df[arg].mean(), inplace=True)
        model_list = ['ARIMA']#'ETS','ZeroesNaive','ETS','GLS','GLM' 'GLS','GLM','LastValueNaive','RollingRegression'
        forecast_length=len(df_data)//4
        model = AutoTS( 
            forecast_length=forecast_length, 
            frequency='infer', 
            prediction_interval=0.90, 
            ensemble=None, 
            max_generations=4, 
            num_validations=2, 
            validation_method='backwards', 
            model_list=model_list, 
            transformer_list='all', 
            models_to_validate=0.2, 
            drop_most_recent=1, 
            n_jobs='auto', 
        )   
        if  '时间' in df_data.columns:
            model = model.fit(df_data, date_col='时间', value_col=arg, id_col=None)
        if  'time' in df_data.columns:
            model = model.fit(df_data, date_col='time', value_col=arg, id_col=None) 
        # prediction = model.predict() 
        # forecasts_df = prediction.forecast
        # validation_results = model.results("validation")
        # prediction.plot(model.df_wide_numeric,
        #         series=model.df_wide_numeric.columns[0],figsize=(10, 6))
        project_path = os.path.split(os.path.realpath(__file__))[0]
            
       # example_filename = "model.csv"  # .csv/.json
        model.export_template(project_path+'\\model\\%s'%modelname, models='best',
                               n=15, max_per_model_class=3)  
        print('model_train Done') 
    except Exception as e:
        log(e)
        print('model_train Fail')                  
                  

def model_forcast(df_data,arg,modelname='model.csv'):##   
    """
    df_data :dataframe
    arg:预测的参数，如水分、黏度等
    modelname：'model.csv'模型信息文件名，
    info:调用模型，预测df_data的arg,返还预测值，包括上下区间
    
    """
    try:    
        project_path = os.path.split(os.path.realpath(__file__))[0]
    
        
        forecast_length=len(df_data)//4
        model = AutoTS(forecast_length=forecast_length,
                                frequency='infer', max_generations=2,
                                num_validations=1, verbose=0)        
        model = model.import_template(project_path+'\\model\\%s'%modelname, method='only') 
        # print("Overwrite template is: {}".format(str(model.initial_template)))
      
        if  '时间' in df_data.columns:
            model = model.fit(df_data, date_col='时间', value_col=arg, id_col=None)
        if  'time' in df_data.columns:
            model = model.fit(df_data, date_col='time', value_col=arg, id_col=None) 
        
        prediction = model.predict() 
        forecasts_df = prediction.forecast 
        forecasts_up, forecasts_low = prediction.upper_forecast, prediction.lower_forecast
        print('model_forcast Done')          
    except Exception as e:
        log(e)
        forecasts_df,forecasts_up, forecasts_low =pd.Series([0]),pd.Series([0]),pd.Series([0])
        print('model_forcast Fail') 
    return  forecasts_df,forecasts_up, forecasts_low
 
def forcast_plt(df_data,arg,modelname='model.csv'):##  
    """
    df_data :dataframe
    arg:预测的参数，如水分、黏度等
    modelname：'model.csv'模型信息文件名，
    info:调用模型，预测df_data的arg,返还预测值，而后进行可视化
    
    """    
    
    try:  
        t1=time.time()
        forecasts_df,forecasts_up, forecasts_low  =model_forcast(df_data,arg,modelname='model.csv')
        t2=time.time()
        print("花费时间:",t2-t1)
        if  '时间' in df_data.columns:
            df_data['时间'] = pd.to_datetime(df_data['时间'])
            df_data=df_data.set_index('时间')  

            temp_df=pd.DataFrame({'origin':df_data[arg],'forcast': forecasts_df[arg],'forecasts_low':forecasts_low[arg],'forecasts_up':forecasts_up[arg]})#,

        if  'time' in df_data.columns:
            df_data['time'] = pd.to_datetime(df_data['time'])
            df_data=df_data.set_index('time')  
            temp_df=pd.DataFrame({'origin':df_data[arg],'forcast': forecasts_df[arg],'forecasts_low':forecasts_low[arg],'forecasts_up':forecasts_up[arg]})#,  
        fig = plt.figure(figsize=(12,5),
                      constrained_layout=True,#类似于tight_layout，使得各子图之间的距离自动调整【类似excel中行宽根据内容自适应】
                     )    
        
        gs = GridSpec(1, 5, figure=fig)                 
        ax1 = fig.add_subplot(gs[0, 0:3])         
    
        for col in temp_df:
            temp_df[col].plot(label=col) 
            plt.legend()
       # temp_df[['origin','forcast','forecasts_low','forecasts_up']].plot()   
        plt.ylabel(arg)
        ax2 = fig.add_subplot(gs[0, 3:5]) 
        for col in temp_df:
            temp_df[col].plot(kind='hist',label=col) 
            plt.legend()
        plt.xlabel(arg)
        plt.savefig(project_path+'\\png\\arma_forcat.png',bbox_inches ='tight') 
        plt.show()    
        print('forcast_plt Done')
    except Exception as e:
        log(e)
        print('forcast_plt Fail')        
        



def predict_judge(df,arg):
    """
    df:dataframe
    arg：和lstm_predict的输入arg保持一致，只用于画纵坐标
    
    原理及作用：
    产生结论，对预测数据进行评价
    return 
        dfoutput:Series ADF检验的结果
        dfcomment:str 诊断语句

    """
    
    try:
        dfoutput,dfcomment=testStationarity(df.index.min(),df.index.max(),arg,df)
        print('lstm_predict_judge Done')
    except Exception as e:    
        log(e)
        print(e)                                             
        dfoutput=None
        dfcomment=None
        print('lstm_predict_judge Fail')        
    return dfoutput,dfcomment



def water_diag1(Series):
    """
    Series：Series
    原理及作用：
        对当前数据的诊断
    return 
        water_tag:诊断标签
        water_comment:str 诊断语句
    """

    water_tag='当前含水量正常'
    water_comment='建议持续关注。'
    with open("./configuration.json", "r", encoding="utf-8") as f:
        content = json.load(f)       
    try:            
        if '水活性' not in Series.index and 'AW' not in Series.index:
            water_tag='未测水分'
            water_comment=''    
        if '水活性' in Series.index:
            if  Series['水活性']> content['AW']['rest'][0]:
                water_tag= content['Conclusion']['water_high'][0]
                water_comment=content['Conclusion']['water_high'][1]
        if 'AW' in  Series.index:
            if  Series['AW']> content['AW']['rest'][0]:
                water_tag= content['Conclusion']['water_high'][0]
                water_comment=content['Conclusion']['water_high'][1]     
        print(water_tag,water_comment) 
        print('water_diag1 Done')
        tag='Done'        
    except Exception as e:
        log(e)
        tag,water_tag,water_comment=e,e,e     
        print('water_diag1 Fail')
    return tag, water_tag,water_comment      
 

def water_diag2(df):
    """
    df：df
    原理及作用：
        对近期水分数据的诊断
    return 
        water_tag:诊断标签
        water_comment:str 诊断语句
    """


    water_tag='含水量趋势状态正常'
    water_comment='建议持续关注。'
    with open("./configuration.json", "r", encoding="utf-8") as f:
        content = json.load(f)       
    try:                
        if '水活性' not in df.columns and 'AW' not in df.columns:
            water_tag='未测水分'
            water_comment=''
        if '水活性' in  df.columns:
            if  df['水活性'].diff(1).sum()>0.2:
                water_tag=content['Conclusion']['water_up'][0]
                water_comment=content['Conclusion']['water_up'][1]
        if 'AW' in  df.columns:
            if  df['AW'].diff(1).sum()>0.2:
                water_tag=content['Conclusion']['water_up'][0]
                water_comment=content['Conclusion']['water_up'][1]  
        print(water_tag,water_comment) 
        print('water_diag2 Done')
        tag='Done'
    except Exception as e:
        log(e)
        tag,water_tag,water_comment=e,e,e
        print('water_diag2 Fail')
    return  tag,water_tag,water_comment     


def water_diag3(df):
    """
    df：df
    原理及作用：
        对未来水分数据的诊断
    return 
        water_tag:诊断标签
        water_comment:str 诊断语句
    """


    water_tag='未来含水量正常'
    water_comment='建议持续关注。'

    with open("./configuration.json", "r", encoding="utf-8") as f:
        content = json.load(f)       
    try:        
        if '水活性' not in df.columns and 'AW' not in df.columns:
            water_tag='未测水分'
            water_comment=''
        if '水活性' in  df.columns:
            if  df['水活性'].diff(1).sum()>0.2:
                water_tag=content['Conclusion']['water_up'][0]
                water_comment=content['Conclusion']['water_up'][1]
        if 'AW' in  df.columns:
            if  df['AW'].diff(1).sum()>0.2:
                water_tag=content['Conclusion']['water_up'][0]
                water_comment=content['Conclusion']['water_up'][1]  
        print(water_tag,water_comment)        
        tag='Done'
        print('water_diag3 Done')
    except Exception as e:
        log(e)
        tag,water_tag,water_comment=e,e,e
        print('water_diag3 Fail')       
    return  tag,water_tag,water_comment     
if __name__ == '__main__':
    plt.rcParams['font.sans-serif']=['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    project_path = os.path.split(os.path.realpath(__file__))[0]
    df_data=pd.read_csv(project_path+'\data\立磨主减速机.csv',encoding='gbk') 
    df_data['时间'] = pd.to_datetime(df_data['时间'])
    a=df_data.resample('1H', on='时间').mean()   
    a=a.reset_index()
    # #####################以下是plan_A####################
    # ######训练模型，得到参数文件####
    # model_train(df_data[1100:1300],'水活性',modelname='model.csv')
    # ######调用模型，进行预测####
    # pred=model_forcast(df_data[1200:1420],'水活性',modelname='model.csv')    
    # #####调用模型，对结果进行可视化####
    # forcast_plt(df_data[1200:1320],'水活性',modelname='model.csv')
    
    
    ######以下是plan_B##############
    #########模型训练、预测和可视化同步进行，不拆分，模型参数进行进行了记录和保留########
    forcast_df=arma_forcast(a[200:300],'黏度',modelname='model.csv')
    #forcast_df=arma_forcast2(a[60:120],'黏度',modelname='model.csv')
    
     ######以下诊断结论测试########
    # water_diag2(df_data)
    # water_diag1(df_data.iloc[1])
    # water_diag3(forecasts_df)    
    # for i in range(len(df_data)-100):
    #     # import time
    #     # tag,validation_results,forecasts_df=forcast_func(df_data.iloc[i:i+100],'水活性')
    #     water_diag2(df_data)
    #     water_diag1(df_data.iloc[1])
    #     # water_diag3(forecasts_df)











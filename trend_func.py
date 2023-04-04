# -*- coding: utf-8 -*-
"""
Created on Mon Jan  9 16:04:06 2023

@author: Lenovo
"""



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from log import log
from ZX_Data_deal import df_to_numd
import os
#from config import Conclusion,para_set
import json
from matplotlib.gridspec import GridSpec
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# global Conclusion
# global para_set


def time_trend_fitcal(df,arg,order=3):
    """info:用于三阶多项式可视化,返还拟合的参数，parameter[0]是三阶参数，parameter[1]是二阶参数，parameter[3]是一阶参数，parameter[0]是0阶参数。
        df:dataframe
        arg；拟合参数，如黏度、水分、
        order:阶次
        横坐标是通过实践计算得到的采样间隔
     
        return: 
            x,y,y2分别失利后的拟合数据
            parameter:拟合的阶数的参数        
    """    
    try:
        if '时间' in df.columns:
            df['时间'] = pd.to_datetime(df['时间'])
            df = df.sort_values(by = '时间')
            df=df.set_index('时间')
        elif 'Time' in df.columns:    
            df['Time'] = pd.to_datetime(df['Time'])
            df = df.sort_values(by = 'Time')
            df=df.set_index('Time')
        else: print('无时间维度，模块报错退出')    
        df=df_to_numd(df)
        df_new=pd.to_numeric(df[arg], errors='coerce').dropna(axis=0)
        #df_new=df[arg].dropna(axis=0)
        if df_new.empty:
            print('数据清洗后为空')
            x,y,y2,parameter=[0],[0],[0],[0]
        else:
            y=df_new.values
            dayZero = df_new.index[0]
            df_new['Day'] = (df_new.index - dayZero).days*24 + (df_new.index - dayZero).seconds/3600
            x=df_new['Day'].values      
            parameter = np.polyfit(x, y, order)
            y2 = sum(parameter[i] * x ** (order-i) for i in range(0,order+1))        
        print('trend_fitcal Done')
        tag='Done'
    except Exception as e:  
        log(e)  
        tag=e
        x,y,y2,parameter=e,e,e,e   
        print('trend_fitcal Fail')
        print(e)         
    return tag,x,y,y2,parameter

def Vis_temp_fitcal(df,arg1='黏度',arg2='油温',order=1):
    """info:三阶多项式拟合
       arg1:拟合y，默认黏度
       arg2:拟合x，默认油温
       order:拟合阶次，默认为1
       return: 
           x,y,y2分别失利后的拟合数据
           parameter:拟合的阶数的参数
    """
    
    try:
        df_new=df[[arg1,arg2]]
        df_new=df_new.sort_values(by = arg2, ascending =True)
        df_new=df_new.dropna(axis=0,how='any')
        if df_new.empty:
            print('数据清洗后为空')
            x,y,y2,parameter=[0],[0],[0],[0]        
        else:        
            y=df_new[arg1].values
            x=df_new[arg2].values
            parameter = np.polyfit(x, y, order) 
            y2 = sum(parameter[i] * x ** (order-i) for i in range(0,order+1))        
        print('Vis_temp_fitcal Done')
        tag='Done'
    except Exception as e:  
        log(e)
        tag,x,y,y2,parameter=e,e,e,e,e       
        print('trend_fitcal Fail')
        print(e)        
    return tag,x,y,y2,parameter   
        
        
        


def trend_plt1(df,arg,order=3):
    """info:用time_trend_fitcal可视化,可视化图片保留在根目录png文件中。
        df:dataframe
        order:拟合阶次，默认为3        
    """
    try: 
        tag,x,y,y2,parameter=time_trend_fitcal(df,arg,order=order)
        if type(x)==str or type(y)==str or type(y2)==str:
            print('计算结果异常，可视化结束')
        else:
            plt.figure(figsize=(10,6),dpi=80)
            plt.scatter(x,y)
            plt.plot(x,y2,lw=3)
            print(parameter)
            title='y='
            _w = []
            for i in range(len(parameter)):
                _w.append(round(float(parameter[i]), 5))
            title = 'y = '
            w = list(map(str, _w))
            for i in range(len(w)):
                if i != 0 and float(w[i]) > 0:
                    w[i] = '+' + w[i]
            for i in range(len(w) - 2):
                title = title + w[i] + '$x^{}$'.format(len(w) - i - 1)
            title = title + w[-2] + '$x$'
            title = title + w[-1]            
            # for i in range(len(parameter)):
            #     title=title+'+'+ '%fx^%ff'%(parameter[i],len(parameter)-i)
            if len(x)!=1:
                #plt.title(np.poly1d(np.polyfit(x,y,deg=order)),fontweight='bold')
                plt.title(title)
            else:pass  
            plt.ylabel('%s'%arg,fontsize=20)
            plt.xlabel('取样间隔/hour',fontsize=20)
            plt.yticks(fontproperties='Times New Roman', size=15,weight='bold')#设置大小及加粗
            plt.xticks(fontproperties='Times New Roman', size=15)
            project_path = os.path.split(os.path.realpath(__file__))[0]
            plt.savefig(project_path+'\\png\\trend.png',bbox_inches = 'tight')
            plt.show()        
        tag='Done'
        print('trend_plt1 Done')
    except Exception as e:  
        log(e)      
        TrendAnalysis=e
        print('trend_plt1 Fail')
        print(e)          
    return tag  



def trend_plt2(df,arg1='黏度',arg2='油温',order=3):
    """info:用于Vis_temp_fitcal的可视化,可视化图片保留在根目录png文件中。
        df:dataframe
        arg1:纵坐标
        arg2:横坐标
        order:拟合阶次，默认为3
    """
    try: 
        tag,x,y,y2,parameter=Vis_temp_fitcal(df,arg1='黏度',arg2='油温',order=order)
        if type(x)==str or type(y)==str or type(y2)==str:print('计算结果异常，可视化结束')
        else:        
            plt.figure(figsize=(10,6),dpi=80)
            plt.scatter(x,y)
            plt.plot(x,y2,lw=3)
            print(parameter)  
            
            title='y='
            _w = []
            for i in range(len(parameter)):
                _w.append(round(float(parameter[i]), 5))
            title = 'y = '
            w = list(map(str, _w))
            for i in range(len(w)):
                if i != 0 and float(w[i]) > 0:
                    w[i] = '+' + w[i]
            for i in range(len(w) - 2):
                title = title + w[i] + '$x^{}$'.format(len(w) - i - 1)
            title = title + w[-2] + '$x$'
            title = title + w[-1]            
            # for i in range(len(parameter)):
            #     title=title+'+'+ '%fx^%ff'%(parameter[i],len(parameter)-i)
            if len(x)!=1:
                #plt.title(np.poly1d(np.polyfit(x,y,deg=order)),fontweight='bold')
                plt.title(title)
            else:pass              
            plt.ylabel('%s'%arg1,fontsize=20)
            plt.xlabel('%s'%arg2,fontsize=20)
            plt.yticks(fontproperties='Times New Roman', size=15,weight='bold')#设置大小及加粗
            plt.xticks(fontproperties='Times New Roman', size=15)
            project_path = os.path.split(os.path.realpath(__file__))[0]
            plt.savefig(project_path+'\\png\\trend.png',bbox_inches = 'tight')
            plt.show()
        tag='Done'
        print('trend_plt2 Done')
    except Exception as e:  
        log(e)     
        tag=e
        print('trend_plt2 Fail')
        print(e)           
    return tag 



def Vis_temp_diag(df,vis):
    """info:对近期(多维数组)黏度、温度及黏温特性的诊断
        vis:黏度等级，用于判断黏度的范围。
        df:dataframe
    """    
    with open("./configuration.json", "r", encoding="utf-8") as f:
       content = json.load(f)  
    if '黏度' in  df.columns:
            
        Vis_tag='黏度无明显变化趋势'
        Vis_comment='建议持续关注'
    else:  
        Vis_tag='未测'
        Vis_comment='未测黏度'
    if '油温' in  df.columns:   
            
        Temp_tag='温度无明显变化趋势'
        Temp_comment='建议持续关注'
    else:  
        Temp_tag='未测'
        Temp_comment='未测油温'  
    if '黏度' in  df.columns and '油温' in  df.columns:   

        Vis_Temp_tag='黏温性能正常'
        Vis_Temp_comment='建议持续关注'
    else:  
        Temp_tag='未知'
        Vis_Temp_comment='未知黏温特性'          
    try:
        tag,x,y,y2,parameter =time_trend_fitcal(df,'黏度',order=1)
        print(parameter[0])      
        if type(parameter[0])==np.float64 and parameter[0]>0.3:
            Vis_tag=content['Conclusion']['Vis_up'][0]
            Vis_comment=content['Conclusion']['Vis_up'][1]
        if type(parameter[0])==np.float64 and parameter[0]<-0.3:
            Vis_tag=content['Conclusion']['Vis_down'][0]   
            Vis_comment=content['Conclusion']['Vis_down'][1]   
        tag,x,y,y2,parameter =time_trend_fitcal(df,'油温',order=1)
        if df['黏度'].iloc[-1]>content['VIS'][str(vis)][0]+content['VIS'][str(vis)][1]:
            Vis_tag=content['Conclusion']['Vis_high'][0]
            Vis_comment=content['Conclusion']['Vis_high'][1]
        print(parameter[0])          
        if type(parameter[0])==np.float64 and parameter[0]>0.3:
            Temp_tag=content['Conclusion']['Temp_up'][0]   
            Temp_comment=content['Conclusion']['Temp_up'][1]   
        tag,x,y,y2,parameter=Vis_temp_fitcal(df,arg1='黏度',arg2='油温',order=1)
        print(parameter[0])      
        if type(parameter[0])==np.float64 and parameter[0]>0.9:
            Vis_Temp_tag=content['Conclusion']['Vis_Temp_unnorm'][0]   
            Vis_Temp_comment=content['Conclusion']['Vis_Temp_unnorm'][1]    
        print(Vis_tag,Vis_Temp_tag)
        tag='Done'
    except Exception as e:  
        log(e)     
        tag,Vis_tag,Vis_comment,Temp_tag,Temp_tag,Vis_Temp_tag,Vis_Temp_comment=e,e,e,e,e,e,e
        print('trend_plt2 Fail')
        print(e)                 
    return   tag,Vis_tag,Vis_comment,Temp_tag,Temp_tag,Vis_Temp_tag,Vis_Temp_comment







def Vis_temp_plt(df,arg1='黏度',arg2='油温',order=3):
    """info:用于Vis_temp_fitcal的可视化,可视化图片保留在根目录png文件中。
        df:dataframe
        arg1:纵坐标
        arg2:横坐标
        order:拟合阶次，默认为3
    """
    try: 
        if '时间' in df.columns:
            df['时间'] = pd.to_datetime(df['时间'])
            df = df.sort_values(by = '时间')
            df=df.set_index('时间')
        elif 'Time' in df.columns:    
            df['Time'] = pd.to_datetime(df['Time'])
            df = df.sort_values(by = 'Time')
            df=df.set_index('Time')        
        fig = plt.figure(figsize=(14,8))
                     #  constrained_layout=True,#类似于tight_layout，使得各子图之间的距离自动调整【类似excel中行宽根据内容自适应】
                     # )    
        plt.subplots_adjust(left=None, bottom=0.15, right=None, top=None,
                            wspace=0.2, hspace=0.45)#wspace 子图横向间距， hspace 代表子图间的纵向距离，left 代表位于图像不同位置
        gs = GridSpec(2, 6, figure=fig) 
                
        ax1 = fig.add_subplot(gs[0, 0:3])         
        df[arg1].plot()
        plt.ylim(df[arg1].min()*0.5, df[arg1].max()*1.5)
        plt.ylabel(arg1)
        
        ax2 = fig.add_subplot(gs[0, 3:6]) 
        df[arg2].plot()
        plt.ylim(df[arg2].min()*0.5, df[arg2].max()*1.5)        
        plt.ylabel(arg2)     
        
        ax3 = fig.add_subplot(gs[1, 0:6]) 
        
        tag,x,y,y2,parameter=Vis_temp_fitcal(df,arg1=arg1,arg2=arg2,order=order)
        if type(x)==str or type(y)==str or type(y2)==str:print('Vis_temp_fitcal计算结果异常，可视化结束')
        else:        

            plt.scatter(x,y)
            #plt.plot(x,y2,lw=3,label='')
            print(parameter)  
            
            title='y='
            _w = []
            for i in range(len(parameter)):
                _w.append(round(float(parameter[i]), 5))
            title = 'y = '
            w = list(map(str, _w))
            for i in range(len(w)):
                if i != 0 and float(w[i]) > 0:
                    w[i] = '+' + w[i]
            for i in range(len(w) - 2):
                title = title + w[i] + '$x^{}$'.format(len(w) - i - 1)
            title = title + w[-2] + '$x$'
            title = title + w[-1]            
            # for i in range(len(parameter)):
            #     title=title+'+'+ '%fx^%ff'%(parameter[i],len(parameter)-i)
            if len(x)!=1:
                #plt.title(np.poly1d(np.polyfit(x,y,deg=order)),fontweight='bold')
                plt.title(title)
                print(type(title))
                plt.plot(x,y2,lw=3,label='%s'%title)
            else:pass              
            plt.ylabel('%s'%arg1,fontsize=20)
            plt.xlabel('%s'%arg2,fontsize=20)
            plt.yticks(fontproperties='Times New Roman', size=15,weight='bold')#设置大小及加粗
            plt.xticks(fontproperties='Times New Roman', size=15)        
        plt.savefig(project_path+'\\png\\Vis_temp_plt.png',bbox_inches ='tight') 
        plt.show()    
    except Exception as e:  
        log(e)     
        tag=e
        print('trend_plt2 Fail')
        print(e)           
    return tag 


if __name__ == '__main__':

    project_path = os.path.split(os.path.realpath(__file__))[0]
    df_data=pd.read_csv(project_path+'\data\华东油气田预测.csv',encoding='gbk') 
    ##测试##
    #trend_plt1(df_data,'黏度',order=3)   
    #Vis_temp_diag(df_data,32)
    Vis_temp_plt(df_data,arg1='黏度',arg2='温度',order=3)
    # for  i in range(1,len(df_data)-200):
    #     print(i)
    #     df=df_data[i:i+200]
    #     Vis_temp_diag(df,46)
    #     trend_plt1(df,'黏度',order=3)   
    #     Vis_temp_fitcal(df,arg1='黏度',arg2='温度',order=3)
    #     trend_plt2(df,arg1='黏度',arg2='温度',order=3) 
        
    
    
    
    
    
    
    
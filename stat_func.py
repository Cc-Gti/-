# -*- coding: utf-8 -*-
"""
Created on Mon Jan  9 15:34:00 2023

@author: Lenovo
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from  ZX_Data_deal import df_to_numd
from log import log
#from config import para_set,Conclusion
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus'] = False
import json

global para_set


#global Conclusion

def cdf_plt(df,arg):
    
    """
    描述：对arg统计,画出CDF曲线及直方分布图,二合一图像
    
    """
    try:           
        hist, bin_edges = np.histogram(df[arg].dropna())#df[np.isfinite(df['distance'])]
        fig=plt.figure(figsize=(11,8))
        #width = (bin_edges[1] - bin_edges[0]) * 0.8
        #plt.bar(bin_edges[1:], hist/max(hist), width=width, color='#5B9BD5')
        #df[i].plot.bar()
        #plt.hist(df[i], color='#5B9BD5')
            # 绘制直方图
        ax1=fig.add_subplot(111)
        plt.rcParams.update({'font.size': 16})#####设置所有字体
        ax1=df[arg].plot(kind = 'hist', bins = bin_edges, color = 'steelblue', edgecolor = 'black', density = True, label = '直方图')
        # 绘制核密度图
        ax1=df[arg].plot(kind = 'kde', color = 'red', label = '核密度估计')
        plt.legend(loc='upper left')
       
        
        plt.xlabel(arg)
       # plt.xlim(0,9)
        ax2 = ax1.twinx()
        cdf = np.cumsum(hist/sum(hist))
        ax2=plt.plot(bin_edges[1:], cdf, '-*', color='#ED7D31',label='累计分布cdf')
        plt.ylabel('Percent')
        plt.legend(loc='upper right')
        project_path = os.path.split(os.path.realpath(__file__))[0]
        plt.savefig(project_path+'\png\cdf_plt.png',bbox_inches = 'tight')
        plt.title(arg)
        plt.show()
        tag='Done'
        print('cdf_plt Done')
    except Exception as e:  
        tag= e
        log(e)
        print(e)
        print('cdf_plt Fail')
    return tag     


def tongji(df):
    """
    描述：对污染、磨损指标进行分布统计,只用于画图
    
    """
    try:               
        plt_list=['4umISO','6umISO','14umISO','21umISO','NAS等级', 'Fe(70~100um)', 'Fe(100~150um)',
                  'Fe(>150um)','nFe(200~300um)','nFe(300~400um)', 'nFe(>400um)']

        # plt_list=['黏度','油温']

        
        
        res = [v for v in plt_list if v in df_data.columns]
        res=df_data[res].dropna(axis=1,how='all').columns.tolist()

        df_new=df[res]
        df_new=df_to_numd(df_new) 

        df_new=df_new.apply(pd.to_numeric, errors='coerce')
        if res==[]:
            tag='no data to visualize' 
        else:
            fig=plt.figure(figsize=(12,12),dpi=200)
            c=0
            for i in range(len(res)):
                c=c+1
                if len(res)<2:
                    ax=fig.add_subplot(1,1,c)
                if len(res)>=2 and len(res)<5:
                    ax=fig.add_subplot(2,2,c)
                elif len(res)>=5 and len(res)<10:
                    ax=fig.add_subplot(3,3,c)
                elif len(res)>=10 and len(res)<17:
                    ax=fig.add_subplot(4,4,c)                        
                ax.hist(df_new[res[i]].dropna(),density=True,) 
                ax.axvline(df_new[res[i]].iloc[-1],color='green')
                plt.subplots_adjust(left=None,bottom=None,right=None,top=None,wspace=0.2,hspace=0.3)
                ax.set_title(r'%s:$\mu=%s$,$\sigma=%s$'%(res[i],round(df_new[res[i]].mean(),2),round(df_new[res[i]].std(),2)),fontsize=9,fontweight='bold')
                plt.yticks(fontproperties='Times New Roman', size=10,weight='bold')#设置大小及加粗
                plt.xticks(fontproperties='Times New Roman', size=10)                
              
      
            project_path = os.path.split(os.path.realpath(__file__))[0]
            plt.savefig(project_path+'\png\Statistics.png',bbox_inches = 'tight')
            plt.show()
            
        print('tongji Done')
        tag='Done'   
    except Exception as e:  
        tag= e
        log(e)
        print(e)
        print('tongji Fail')
    return tag               


def wear_nas_diag(Series,facility_type):
    """
    描述：对当前结果(一维数组)磨损指标进行诊断，输出诊断结论和语句
    Series：dataframe.series类
    facility_type：['齿轮箱','汽轮机','液压系统','压缩机']
    return 诊断标签和语句
    """   
    with open("./configuration.json", "r", encoding="utf-8") as f:
       content = json.load(f)       
    list1=['Fe(70~100um)','Fe(100~150um)','Fe(>150um)','总铁磁颗粒','铁磁细颗粒','铁磁粗颗粒']
    if [v for v in list1 if v in Series.index]!=[]:
        Fe_tag='当前铁磁颗粒磨损情况正常'
        Fe_commnent='建议持续关注。'
    else:
        Fe_tag='未测'
        Fe_commnent='未测铁磁颗粒含量。'        
    list2=['nFe(200~300um)','nFe(300~400um)','nFe(>400um)','非铁磁颗粒(200um~300um)','非铁磁颗粒(300um~400um)','非铁磁颗粒(400um~)']
    if [v for v in list2 if v in Series.index]!=[]:        
        nFe_tag='当前非铁磁颗粒磨损情况正常'
        nFe_commnent='建议持续关注。' 
    else:
        nFe_tag='未测'
        nFe_commnent='未测铁磁颗粒含量。' 
    list3=['NAS等级']    
    if [v for v in list3 if v in Series.index]!=[]:        
        NAS_tag='当前污染情况正常'
        NAS_comment='建议持续关注。' 
    else:
        NAS_tag='未测'
        NAS_comment='未测污染度。'         
    try:           
        if facility_type=='汽轮机':
            if 'Fe(70~100um)' in Series.index or '铁磁细颗粒' in  Series.index:
                if Series['Fe(70~100um)']>content['Fe_Turbine']['Fe(70~100um)'][0]:
                    Fe_tag=content['Conclusion']['Fe_high'][0]
                    Fe_commnent=content['Conclusion']['Fe_high'][1]
            if 'Fe(100~150um)' in Series.index or '铁磁粗颗粒' in  Series.index: 
                if Series['Fe(100~150um)']>content['Fe_Turbine']['Fe(100~150um)'][0]:
                    Fe_tag=content['Conclusion']['Fe_high'][0]
                    Fe_commnent=content['Conclusion']['Fe_high'][1]     
            if 'Fe(>150um)' in Series.index or '总铁磁颗粒' in  Series.index:
                if Series['Fe(>150um)']>content['Fe_Turbine']['Fe(>150um)'][0]:
                    Fe_tag=content['Conclusion']['Fe_high'][0]
                    Fe_commnent=content['Conclusion']['Fe_high'][1]      
            if 'nFe(200~300um)' in Series.index or '非铁磁颗粒(200um~300um)' in Series.index:
                if Series['nFe(200~300um)']>content['Fe_Turbine']['nFe(200~300um)'][0]:
                    nFe_tag=content['Conclusion']['nFe_high'][0]
                    nFe_commnent=content['Conclusion']['nFe_high'][1]          
            if 'nFe(300~400um)' in Series.index or '非铁磁颗粒(300~400um)' in Series.index:
                if Series['nFe(300~400um)']>content['Fe_Turbine']['nFe(300~400um)'][0]:
                    nFe_tag=content['Conclusion']['nFe_high'][0]
                    nFe_commnent=content['Conclusion']['nFe_high'][1] 
            if 'nFe(>400um)' in Series.index or '非铁磁颗粒(400um~)' in Series.index:
                if Series['nFe(>400um)']>content['Fe_Turbine']['nFe(>400um)'][0]:
                    nFe_tag=content['Conclusion']['nFe_high'][0]
                    nFe_commnent=content['Conclusion']['nFe_high'][1]        
        if facility_type=='液压系统':
            if 'Fe(70~100um)' in Series.index or '铁磁细颗粒' in  Series.index:
                if Series['Fe(70~100um)']>content['Fe_Hydraulic']['Fe(70~100um)'][0]:
                    Fe_tag=content['Conclusion']['Fe_high'][0]
                    Fe_commnent=content['Conclusion']['Fe_high'][1]
            if 'Fe(100~150um)' in Series.index or '铁磁粗颗粒' in  Series.index: 
                if Series['Fe(100~150um)']>content['Fe_Hydraulic']['Fe(100~150um)'][0]:
                    Fe_tag=content['Conclusion']['Fe_high'][0]
                    Fe_commnent=content['Conclusion']['Fe_high'][1]     
            if 'Fe(>150um)' in Series.index or '总铁磁颗粒' in  Series.index: 
                if Series['Fe(>150um)']>content['Fe_Hydraulic']['Fe(>150um)'][0]:
                    Fe_tag=content['Conclusion']['Fe_high'][0]
                    Fe_commnent=content['Conclusion']['Fe_high'][1]      
            if 'nFe(200~300um)' in Series.index or '非铁磁颗粒(200um~300um)' in Series.index:
                if Series['nFe(200~300um)']>content['Fe_Hydraulic']['nFe(200~300um)'][0]:
                    nFe_tag=content['Conclusion']['nFe_high'][0]
                    nFe_commnent=content['Conclusion']['nFe_high'][1]          
            if 'nFe(300~400um)' in Series.index or '非铁磁颗粒(300~400um)' in Series.index:
                if Series['nFe(300~400um)']>content['Fe_Hydraulic']['nFe(300~400um)'][0]:
                    nFe_tag=content['Conclusion']['nFe_high'][0]
                    nFe_commnent=content['Conclusion']['nFe_high'][1] 
            if 'nFe(>400um)' in Series.index or '非铁磁颗粒(400um~)' in Series.index:
                if Series['nFe(>400um)']>content['Fe_Hydraulic']['nFe(>400um)'][0]:
                    nFe_tag=content['Conclusion']['nFe_high'][0]
                    nFe_commnent=content['Conclusion']['nFe_high'][1]                          
        if facility_type=='压缩机':
            if 'Fe(70~100um)' in Series.index or '铁磁细颗粒' in  Series.index:
                if Series['Fe(70~100um)']>content['Fe_Hydraulic']['Fe(70~100um)'][0]:
                    Fe_tag=content['Conclusion']['Fe_high'][0]
                    Fe_commnent=content['Conclusion']['Fe_high'][1]
            if 'Fe(100~150um)' in Series.index or '铁磁粗颗粒' in  Series.index: 
                if Series['Fe(100~150um)']>content['Fe_Hydraulic']['Fe(100~150um)'][0]:
                    Fe_tag=content['Conclusion']['Fe_high'][0]
                    Fe_commnent=content['Conclusion']['Fe_high'][1]     
            if 'Fe(>150um)' in Series.index or '总铁磁颗粒' in  Series.index:
                if Series['Fe(>150um)']>content['Fe_Hydraulic']['Fe(>150um)'][0]:
                    Fe_tag=content['Conclusion']['Fe_high'][0]
                    Fe_commnent=content['Conclusion']['Fe_high'][1]      
            if 'nFe(200~300um)' in Series.index or '非铁磁颗粒(200um~300um)' in Series.index:
                if Series['nFe(200~300um)']>content['Fe_Hydraulic']['nFe(200~300um)'][0]:
                    nFe_tag=content['Conclusion']['nFe_high'][0]
                    nFe_commnent=content['Conclusion']['nFe_high'][1]          
            if 'nFe(300~400um)' in Series.index or '非铁磁颗粒(300~400um)' in Series.index:
                if Series['nFe(300~400um)']>content['Fe_Hydraulic']['nFe(300~400um)'][0]:
                    nFe_tag=content['Conclusion']['nFe_high'][0]
                    nFe_commnent=content['Conclusion']['nFe_high'][1] 
            if 'nFe(>400um)' in Series.index or '非铁磁颗粒(400um~)' in Series.index:
                if Series['nFe(>400um)']>content['Fe_Hydraulic']['nFe(>400um)'][0]:
                    nFe_tag=content['Conclusion']['nFe_high'][0]
                    nFe_commnent=content['Conclusion']['nFe_high'][1]      
        if facility_type=='齿轮箱':
            if 'Fe(70~100um)' in Series.index or '铁磁细颗粒' in  Series.index:
                if Series['Fe(70~100um)']>content['Fe_gear']['Fe(70~100um)'][0]:
                    Fe_tag=content['Conclusion']['Fe_high'][0]
                    Fe_commnent=content['Conclusion']['Fe_high'][1]
            if 'Fe(100~150um)' in Series.index or '铁磁粗颗粒' in  Series.index: 
                if Series['Fe(100~150um)']>content['Fe_gear']['Fe(100~150um)'][0]:
                    Fe_tag=content['Conclusion']['Fe_high'][0]
                    Fe_commnent=content['Conclusion']['Fe_high'][1]     
            if 'Fe(>150um)' in Series.index or '总铁磁颗粒' in  Series.index:
                if Series['Fe(>150um)']>content['Fe_gear']['Fe(>150um)'][0]:
                    Fe_tag=content['Conclusion']['Fe_high'][0]
                    Fe_commnent=content['Conclusion']['Fe_high'][1]      
            if 'nFe(200~300um)' in Series.index or '非铁磁颗粒(200um~300um)' in Series.index:
                if Series['nFe(200~300um)']>content['Fe_gear']['nFe(200~300um)'][0]:
                    nFe_tag=content['Conclusion']['nFe_high'][0]
                    nFe_commnent=content['Conclusion']['nFe_high'][1]          
            if 'nFe(300~400um)' in Series.index or '非铁磁颗粒(300~400um)' in Series.index:
                if Series['nFe(300~400um)']>content['Fe_gear']['nFe(300~400um)'][0]:
                    nFe_tag=content['Conclusion']['nFe_high'][0]
                    nFe_commnent=content['Conclusion']['nFe_high'][1] 
            if 'nFe(>400um)' in Series.index or '非铁磁颗粒(400um~)' in Series.index:
                if Series['nFe(>400um)']>content['Fe_gear']['nFe(>400um)'][0]:
                    nFe_tag=content['Conclusion']['nFe_high'][0]
                    nFe_commnent=content['Conclusion']['nFe_high'][1] 
        if 'NAS等级' in Series.index:
            if Series['NAS等级']>content['NAS']['All'][0]:
                NAS_tag=content['Conclusion']['NAS_high'][0]
                NAS_comment=content['Conclusion']['NAS_high'][1]
        tag= 'Done'
    except Exception as e:  
        tag= e
        log(e)
        NAS_tag,NAS_comment,Fe_tag,Fe_commnent,nFe_tag,nFe_commnent=e,e,e,e,e,e
        print(e)
        print('wear_nas_diag Fail')
                    
    print(Fe_tag,Fe_commnent,nFe_tag,nFe_commnent)            
    return tag,NAS_tag,NAS_comment,Fe_tag,Fe_commnent,nFe_tag,nFe_commnent      


def wear_nas_diag2(df,facility_type):
    """
    描述：对近期结果(多维数组)的磨损指标进行诊断，输出诊断结论和语句
    Series：dataframe.series类
    facility_type：['齿轮箱','汽轮机','液压系统','压缩机']
    return 诊断标签和语句
    """       
    with open("./configuration.json", "r", encoding="utf-8") as f:
       content = json.load(f)     
    if '时间' in df.columns: 
        df['时间'] = pd.to_datetime(df['时间'])        
        df=df.set_index('时间') 
    if 'time' in df.columns: 
        df['time'] = pd.to_datetime(df['time'])        
        df=df.set_index('time')         
    df=df.sort_index()        
    list1=['Fe(70~100um)','Fe(100~150um)','Fe(>150um)','总铁磁颗粒','铁磁细颗粒','铁磁粗颗粒']
    if [v for v in list1 if v in df.columns]!=[]:
        Fe_tag='近期铁磁颗粒磨损情况正常'
        Fe_comment='建议持续关注。'
    else:
        #print([v for v in list1 if v in df.columns])
        Fe_tag='未测'
        Fe_comment='未测铁磁颗粒含量。'        
    list2=['nFe(200~300um)','nFe(300~400um)','nFe(>400um)','非铁磁颗粒(200um~300um)','非铁磁颗粒(300um~400um)','非铁磁颗粒(400um~)']
    if [v for v in list2 if v in df.columns]!=[]:        
        nFe_tag='近期非铁磁颗粒磨损情况正常'
        nFe_comment='建议持续关注。' 
    else:
        nFe_tag='未测'
        nFe_comment='未测铁磁颗粒含量。' 
    list3=['NAS等级']
    if [v for v in list3 if v in df.columns]!=[]:
        NAS_tag='近期污染情况正常'
        NAS_comment='建议持续关注。'
    else:
        #print([v for v in list1 if v in df.columns])
        NAS_tag='未测'
        NAS_comment='未测污染度。'                  
    # if 'time' in df.columns:
    #     df['time'] = pd.to_datetime(df['time'])  
    #     df=df.set_index('time')
    #print(df[[v for v in list1 if v in df.columns]].diff(1).sum())  
    try:    
        if facility_type=='汽轮机':
            if  (df[[v for v in list1 if v in df.columns]].diff(20).max()> 10).any():##暂定
                Fe_tag=content['Conclusion']['Fe_up'][0]
                Fe_comment=content['Conclusion']['Fe_up'][1]   
            if  (df[[v for v in list2 if v in df.columns]].diff(20).max()>10).any():
                nFe_tag=content['Conclusion']['nFe_up'][0]
                nFe_comment=content['Conclusion']['nFe_up'][1]
            if  (df[[v for v in list3 if v in df.columns]].diff(20).max()>3).any():
                NAS_tag=content['Conclusion']['NAS_up'][0]
                NAS_comment=content['Conclusion']['NAS_up'][1]    

        if facility_type=='液压系统':   
            if  (df[[v for v in list1 if v in df.columns]].diff(20).max()>10).any():
                Fe_tag=content['Conclusion']['Fe_up'][0]
                Fe_comment=content['Conclusion']['Fe_up'][1]      
            if  (df[[v for v in list2 if v in df.columns]].diff(20).max()>10).any():
                nFe_tag=content['Conclusion']['nFe_up'][0]
                nFe_comment=content['Conclusion']['nFe_up'][1]
            if  (df[[v for v in list3 if v in df.columns]].diff(20).max()>3).any():
                NAS_tag=content['Conclusion']['NAS_up'][0]
                NAS_comment=content['Conclusion']['NAS_up'][1]    

        if facility_type=='压缩机':
            if  (df[[v for v in list1 if v in df.columns]].diff(20).max()>10).any():
                Fe_tag=content['Conclusion']['Fe_up'][0]
                Fe_comment=content['Conclusion']['Fe_up'][1]      
            if  (df[[v for v in list2 if v in df.columns]].diff(20).max()>10).any():
                nFe_tag=content['Conclusion']['nFe_up'][0]
                nFe_comment=content['Conclusion']['nFe_up'][1]
            if  (df[[v for v in list3 if v in df.columns]].diff(20).max()>3).any():
                NAS_tag=content['Conclusion']['NAS_up'][0]
                NAS_comment=content['Conclusion']['NAS_up'][1]    
        if facility_type=='齿轮箱':
            if  (df[[v for v in list1 if v in df.columns]].diff(20).max()>100).any():
                print(df[[v for v in list1 if v in df.columns]].diff(10).max())
                Fe_tag=content['Conclusion']['Fe_up'][0]
                Fe_comment=content['Conclusion']['Fe_up'][1]       
            if  (df[[v for v in list2 if v in df.columns]].diff(20).max()>100).any():
                nFe_tag=content['Conclusion']['nFe_up'][0]
                nFe_comment=content['Conclusion']['nFe_up'][1]
            if  (df[[v for v in list3 if v in df.columns]].diff(20).sum()>3).any():
                NAS_tag=content['Conclusion']['NAS_up'][0]
                NAS_comment=content['Conclusion']['NAS_up'][1]              
        tag='Done'
        print('wear_nas_diag2 Done') 
    except Exception as e:  
        tag= e
        log(e)
        print(e)
        NAS_tag,NAS_comment,Fe_tag, Fe_comment ,nFe_tag,nFe_comment=e,e,e,e,e,e
        print('wear_nas_diag2 Fail')        
    return  tag,NAS_tag,NAS_comment,Fe_tag, Fe_comment ,nFe_tag,nFe_comment      
 

if __name__ == '__main__':
    project_path = os.path.split(os.path.realpath(__file__))[0]
    df_data=pd.read_csv(project_path+'\data\立磨主减速机.csv',encoding='gbk') 
 
    # ###以下是测试###
    for  i in range(1,len(df_data)-400):
        df=df_data[i:i+500]
        #DataStatistic=tongji(df)
        tongji(df)
        tag,NAS_tag,NAS_comment,Fe_tag, Fe_comment ,nFe_tag,nFe_comment =wear_nas_diag(df.iloc[-1],facility_type='齿轮箱')
        tag,NAS_tag,NAS_comment,Fe_tag, Fe_comment ,nFe_tag,nFe_comment =wear_nas_diag2(df,facility_type='齿轮箱')
            
            
            
            
            
            
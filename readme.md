算法说明文档
=========

##算法库v1.0:


    ├── ReadMe.md           // 帮助文档
    
    ├── Data           // 数据源模板文件
    
    ├── log          // 内部为运行bug日志
    
    ├── png           // 运行完成后的结果图像   
    
    ├── result           // 自动生成分析报告的路径 
    
    ├── config.py   // 参数配置文件，目前主要内容为诊断专家知识
    ├── stat_func.py   // 用于黏温特性拟合
    ├── arma_func.py   // arma时间序列预测,参数选择为遗传算法，过程较慢，
    ├── keras_lstm.py   // lstm时间序列预测,基于tensorflow，只能用于64位系统。
    ├── log.py   // bug日志的生成脚本
    ├── stat_func.py   // 用于污染及磨损统计
    ├── zaixian.py   // 包含统计、可视化、STL算法
    ├── ZX_Data_deal.py   // 数据处理
    ├── ZX_report.py   // 智能生成总结分析报告
    
    ├── requirement.txt   // 只用于包的版本查询

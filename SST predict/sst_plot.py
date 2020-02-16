'''
@Author: your name
@Date: 2020-02-16 10:10:16
@LastEditTime: 2020-02-16 10:17:44
@LastEditors: Please set LastEditors
@Description: In User Settings Edit
@FilePath: \MCM_2020_A\SST predict\sst_plot.py
'''
import numpy as np 
import matplotlib
import matplotlib.pyplot as plt

# 从plot文件夹导入plot文件
import plot.heatMapPlot as pt
# 导入数据
import sst_data

if __name__ == "__main__":
    # 初始化数据
    data_prdict = sst_data.latestData
    # 载入斜率数据集
    slopeList =  np.loadtxt("SST predict/slopeList_.txt")
    
    '''
    train step	lossRate	slope	train loss	
    2000	    0.15	    0.0756  0.1072	  最坏
    2000	    0.2	        0.0566	0.082	  最可能
    2000	    0.4	        0.0377	0.12	  最好
    '''

    # n年后的温度值
    yer = 10
    data_prdict += yer * slopeList * (377 / 377)  # 乘以最好，最坏，正常的比率
    
    sst = np.array(data_prdict)
    longitude = range(65, 49, -1)
    latitude = range(10, -4, -1)

    fig, ax = plt.subplots()
    im, cbar = pt.heatmap(sst, longitude, latitude, ax=ax,
                    cmap=plt.cm.hot_r, cbarlabel="sea surface temperature after" +  str(yer)  + "years (0.00 means lands) [c]")
    texts = pt.annotate_heatmap(im, valfmt="{x:.1f}c")

    fig.tight_layout()
    plt.show()

    



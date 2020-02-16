import tensorflow.compat.v1 as tf
import numpy as np
import matplotlib.pyplot as plt
import csv
import sst_data

def readCsv(filename):
    File = open("SST predict/" + filename, "r")
    Reader = csv.reader(File)    

    # 建立空数组
    Time, Data = [], []
    for item in Reader:
        # 忽略第一行
        if Reader.line_num == 1:
            continue
        try:
            Time.append(float(item[0]))
            Data.append(float(item[2]))
        except BaseException as e:
            break
    return Time, Data



def linearExgression(testTime, testData, checkTime, checkData, lossRate, ifPlot):
    '''
    @description: 线性回归
    @param {testTime, testData, 0(不测试), 0(不测试), lossRate(损失函数参数) , true(是否绘图)} 
    @return: {最大值， 最小值， 两者距离}
    '''
    #设置参数
    #设置梯度下降算法的学习率
    learning_rate = 0.01
    #设置迭代次数
    max_steps = 2000
    #每迭代100次输出一次loss
    show_step = 100
    # 模拟训练数据
    train_X = np.asarray(testTime)
    train_Y = np.asarray(testData)
    #获取训练数据的大小
    n_samples = train_X.shape[0]
    #定义输入变量,变量只有一个特征
    X = tf.placeholder(dtype=tf.float32)
    #定义输出变量,输出只有一个值
    Y = tf.placeholder(dtype=tf.float32)
 
    #设计模型
    #定义权重
    rng = np.random
    W = tf.Variable(rng.randn(),name="weight")
    #定义偏置
    b = tf.Variable(rng.randn(),name="bias")
    #计算预测值
    pred = tf.add(tf.multiply(X,W),b)
 
    #定义损失函数
    loss = lossRate * tf.reduce_sum(tf.pow(pred-Y,2)) / n_samples
    #使用梯度下降算法来优化损失函数
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
 
    #创建一个会话
    with tf.Session() as sess:
        # 初始化所有变量
        initialize = tf.global_variables_initializer().run()
        #迭代训练
        for step in range(max_steps):
            for (x,y) in zip(train_X,train_Y):
                sess.run(train_step,feed_dict={X:x,Y:y})
            if step % show_step == 0:
                #计算模型在数据集上的损失
                step_loss = sess.run([loss],feed_dict={X:train_X,Y:train_Y})
                print("step:",step,"-step loss:%.4f",step_loss)
        #计算最终的Loss
        train_loss = sess.run([loss],feed_dict={X:train_X,Y:train_Y})
        print("train loss:%.4f",train_loss)
        #输出参数
        print("weights:",sess.run(W),"-bias:",sess.run(b))

        arrayPredict = sess.run(pred,feed_dict={X:train_X})

        plt.plot(train_X, train_Y,"ro", label="original data")
        plt.plot(train_X, arrayPredict, label="predict data")
        plt.legend(loc="upper left")
        if ifPlot == True:
            plt.savefig("./" + str(lossRate) + ".png")
            # plt.show()
        else:
            pass

        # #测试数据
        # test_X = np.asarray(checkTime)
        # test_Y = np.asarray(checkData)
        # #计算回归模型在测试数据上的loss
        # print("test loss:%.4f",sess.run(loss,feed_dict={X:test_X,Y:test_Y}))
        maxVal, minVal, disVal = max(arrayPredict), min(arrayPredict), arrayPredict.size
        return maxVal, minVal, disVal
    pass


def curveFitting(testTime, testData):
    '''
    @description: 曲线拟合
    @param {type} 
    @return: none
    '''
    a, b = [], []
    for i in range(len(testTime)):
        a.append([testTime[i]])
    for i in range(len(testData)):
        b.append([testData[i]])
    
    x_data = np.asarray(a)
    y_data = np.asarray(b)
    x = tf.placeholder(tf.float32, [None, 1])  # 定义占位符
    y = tf.placeholder(tf.float32, [None, 1])
    
    # 定义输入层
    weight_1 = tf.Variable(tf.random_normal([1, 10]))  # 权重矩阵为1*10，即1个输入，10个中间层
    biase_1 = tf.Variable(tf.zeros([1, 10]))  # 偏置值
    wx_plus_1 = tf.matmul(x, weight_1) + biase_1  # 输入数据与权值相乘
    L1 = tf.nn.tanh(wx_plus_1)  # 激活函数
    
    # 定义输出层
    weight_2 = tf.Variable(tf.random_normal([10, 1]))  # 10个中间层，1个输出层
    biase_2 = tf.Variable(tf.zeros([1, 1]))
    wx_plus_2 = tf.matmul(L1, weight_2) + biase_2
    prediction = tf.nn.tanh(wx_plus_2)
    
    loss = tf.reduce_mean(tf.square(y - prediction))  # 求每一个数据误差的平方，再求均值
    train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)  # 梯度下降法优化器，学习率为0.1，使得误差最小
    with tf.Session() as sees:
        sees.run(tf.global_variables_initializer())  # 变量初始化
        for i in range(1000):
            sees.run(train_step, feed_dict={x: x_data, y: y_data})  # 进行2000次训练
    
        prediction_value = sees.run(prediction, feed_dict={x: x_data})  # 预测
        plt.figure()  # 画图
        plt.scatter(x_data, y_data)  # 画输入点
        plt.plot(x_data, prediction_value, 'r-', lw=5)  # 画预测曲线
        plt.show()

    pass


def predictPlot(initVal, slope):
    '''
    @description: 预测以后50年内的温度图像
    @param ：none
    @return: none
    '''
    x = range(2020, 2070)
    y = [item * slope + initVal for item in range(1, 51)]
    pass

def testFunc():

    testTime, testData = readCsv("srcdata.csv")
    checkTime, checkData = readCsv("testdata.csv")

    maxVal, minVal, disVal = linearExgression(testTime, testData, 0, 0, 0.15, True)
    # curveFitting(testTime, testData)

    # 求得斜率
    slope = (maxVal - minVal) / disVal
    print("slope : ", slope)
    predictPlot(maxVal, slope)
    pass

if __name__ == "__main__":
    tf.disable_v2_behavior()
    # testFunc()

    # 斜率的统计表
    slopeList  = [[0]*14 for i in range(16)]


    for lossRate in [0.15, 0.20, 0.55]:
        # 定位数据点
        # 循环同一纬度
        for lat in range(0, 16):
            # 循环同一经度
            for lon in range(0, 14):  
                testData = []
                # 同样经度/纬度，不同年份的数据
                for yer in range(0, int(len(sst_data.data) / 16)):
                    testData.append(sst_data.data[yer * 16 + lat][lon])
                # print(testData) # 测试是否遍历成功

                maxVal, minVal, disVal = linearExgression(range(0, 48), testData, 0, 0, lossRate, False)
                slopeList[lat][lon] = (maxVal - minVal) / disVal
        
        # 保存文件
        np.savetxt("SST predict/slopeList_" + str(lossRate * 100) + ".txt",slopeList)
    


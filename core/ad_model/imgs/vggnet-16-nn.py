# -*- coding: utf-8 -*-

from datetime import datetime
import math
import time
import tensorflow as tf
from PIL import Image
import os
import numpy as np
import pandas as pd
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
import json
from sklearn import preprocessing
from PIL import Image
import matplotlib as plt

'''
网络名称：VGGNet16-D 



卷积层创建函数，并将本层参数存入参数列表
input_op:输入的tensor  name:这一层的名称  kh:kernel height即卷积核的高    kw:kernel width即卷积核的宽
n_out:卷积核数量即输出通道数   dh:步长的高     dw:步长的宽     p:参数列表
'''


temp_path              =     "D:\cjh-model\AD-Model\AD-ADNI-DATA\clinical_data\data_temp\\"
path_224               =     "D:\cjh-model\AD-Model\AD-ADNI-DATA\image_data_224\\"
path_224_train         =     "D:\cjh-model\AD-Model\AD-ADNI-DATA\image_data_224_train\\"
path_224_test          =     "D:\cjh-model\AD-Model\AD-ADNI-DATA\image_data_224_test\\"
path_800               =     "D:\cjh-model\AD-Model\AD-ADNI-DATA\image_data_800\\"
path_224_test_1        =     "D:\cjh-model\AD-Model\AD-ADNI-DATA\image_data_224_test_1\\"
path_model             =     "D:\cjh-model\AD-Model\AD-ADNI-DATA\\"


BATCH_SIZE = 13  # 一个批次的数据
num_batches = 20  # 测试一百个批次的数据
learning_rate=0.001 #学习率

#
NUM_CLASS = 3
IMAGE_HEIGHT = 224
IMAGE_WIDTH = 224
IMAGE_CHANNELS = 3 # 彩色=3，灰白=1

L_CN = [[1, 0, 0]]
L_MCI = [[0, 1, 0]]
L_AD = [[0, 0, 1]]


fc8_temp_1_x = []
fc8_temp_1_y = np.random.rand(1, 3)
fc8_temp_1_y = np.delete(fc8_temp_1_y, 0, axis=0)


fc8_temp_2_x = []
fc8_temp_2_y = np.random.rand(1, 3)
fc8_temp_2_y = np.delete(fc8_temp_2_y, 0, axis=0)

fc8_temp_3_x = []
fc8_temp_3_y = np.random.rand(1, 3)
fc8_temp_3_y = np.delete(fc8_temp_3_y, 0, axis=0)

fc8_merge_x_train = []
fc8_merge_x_test = []

fc8_merge_y_train = np.random.rand(1, 3)
fc8_merge_y_train = np.delete(fc8_merge_y_train, 0, axis=0)

fc8_merge_y_test = np.random.rand(1, 3)
fc8_merge_y_test = np.delete(fc8_merge_y_test, 0, axis=0)

max_epoch = 0
max_batch = 0





def gpu_config():
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.8
    set_session(tf.Session(config=config))
    
# 调整图像大小,将800x800转换成224x224,并将4通道图像转换成3通道图像
def change_image():
    image_count = 0
    listPngFileName = os.listdir(path_800)
    for pngFileName in listPngFileName:
        image_count = image_count + 1
        image_count = image_count + 1
        png_image = Image.open(path_800 + pngFileName)
        #print(type(png_image.size))
        # 将in_image图像转换成 256 x 256 规格
        r, g, b, a = png_image.split()
        out_image = Image.merge("RGB", (r, g, b))
        out_image = out_image.resize((224, 224), Image.ANTIALIAS)
        out_image.save(path_224 + pngFileName)
    print("图像总共的数量为：")
    print(image_count)


# vggnet-16训练集划分多少张图片
def divide_data_1(image_num):
    image_count_1 = 0
    image_count_2 = 0
    image_count_3 = 0

    listPngFileName = os.listdir(path_800)
    for pngFileName in listPngFileName:

        fileName = pngFileName.split(".")
        nameInfo = fileName[0].split("-")
        if(nameInfo[1] == "1" and image_count_1 < image_num):
            image_count_1 = image_count_1 + 1
            png_image = Image.open(path_224 + pngFileName)
            png_image.save(path_224_train + pngFileName)
        if(nameInfo[1] == "1" and image_count_1 >= image_num):
            png_image = Image.open(path_224 + pngFileName)
            png_image.save(path_224_test + pngFileName)
        if(nameInfo[1] == "2" and image_count_2 < image_num):
            image_count_2 = image_count_2 + 1
            png_image = Image.open(path_224 + pngFileName)
            png_image.save(path_224_train + pngFileName)
        if(nameInfo[1] == "2" and image_count_2 >= image_num):
            png_image = Image.open(path_224 + pngFileName)
            png_image.save(path_224_test + pngFileName)
        if(nameInfo[1] == "3" and image_count_3 < image_num):
            image_count_3 = image_count_3 + 1
            png_image = Image.open(path_224 + pngFileName)
            png_image.save(path_224_train + pngFileName)
        if(nameInfo[1] == "3" and image_count_3 >= image_num):
            png_image = Image.open(path_224 + pngFileName)
            png_image.save(path_224_test + pngFileName)


def divide_data_2(num):
    for i in range(302):
        array = np.full((1,1000),0.7)
        array_1_x = fc8_temp_1_x[i]
        array_2_x = fc8_temp_2_x[i]
        array_3_x = fc8_temp_3_x[i]
        global fc8_merge_y_train
        global fc8_merge_y_test

        if(i < num):
            if((fc8_temp_1_y[i,:] == np.array(L_CN).reshape(1, 3)).all()):
                fc8_merge_x_train.append(np.hstack((array_1_x + array, array_2_x, array_3_x)))
            if ((fc8_temp_1_y[i, :] == np.array(L_MCI).reshape(1, 3)).all()):
                fc8_merge_x_train.append(np.hstack((array_1_x, array_2_x + array, array_3_x)))
            if ((fc8_temp_1_y[i, :] == np.array(L_AD).reshape(1, 3)).all()):
                fc8_merge_x_train.append(np.hstack((array_1_x, array_2_x, array_3_x + array)))
        if(i >= num):
            if ((fc8_temp_1_y[i, :] == np.array(L_CN).reshape(1, 3)).all()):
                fc8_merge_x_test.append(np.hstack((array_1_x + array, array_2_x, array_3_x)))
            if ((fc8_temp_1_y[i, :] == np.array(L_MCI).reshape(1, 3)).all()):
                fc8_merge_x_test.append(np.hstack((array_1_x, array_2_x + array, array_3_x)))
            if ((fc8_temp_1_y[i, :] == np.array(L_AD).reshape(1, 3)).all()):
                fc8_merge_x_test.append(np.hstack((array_1_x, array_2_x, array_3_x + array)))

        if(i < num):
            fc8_merge_y_train = np.row_stack((fc8_merge_y_train, fc8_temp_1_y[i,:]))
        if(i >= num):
            fc8_merge_y_test = np.row_stack((fc8_merge_y_test, fc8_temp_1_y[i,:]))


def divide_data_3(num):
    for i in range(302):
        array_1_x = fc8_temp_1_x[i]
        array_2_x = fc8_temp_2_x[i]
        array_3_x = fc8_temp_2_x[i]

        global fc8_merge_y_train
        global fc8_merge_y_test

        if (i < num):
            fc8_merge_x_train.append(array_1_x + array_2_x + array_3_x)
        if (i >= num):
            fc8_merge_x_test.append(array_1_x + array_2_x + array_3_x)

        if (i < num):
            fc8_merge_y_train = np.row_stack((fc8_merge_y_train, fc8_temp_1_y[i, :]))
        if (i >= num):
            fc8_merge_y_test = np.row_stack((fc8_merge_y_test, fc8_temp_1_y[i, :]))


# 读取图像
'''

'''
def read_data_1(position,batch_size, batch,target):
    datas = []
    image_count = 0
    labels = np.random.rand(1, 3)
    labels = np.delete(labels, 0, axis=0)
    if (target == 1):
        listPngFileName = os.listdir(path_224_train)
        for pngFileName in listPngFileName:
            fileName = pngFileName.split(".")
            nameInfo = fileName[0].split("-")
            if (nameInfo[1] == str(position)):
                # print(str(image_count) + " - " + str(batch_size*batch) + " - " + str(batch_size*(batch + 1)))
                image_count = image_count + 1
                if(image_count > batch_size*batch and image_count <= batch_size*(batch + 1)):
                    #print(pngFileName)
                    png_image = Image.open(path_224_train + pngFileName)
                    datas.append(np.array(png_image)/255.0)
                    if (nameInfo[2] is "1"):
                        label = np.array(L_CN).reshape(1, 3)
                    if (nameInfo[2] is "2"):
                        label = np.array(L_MCI).reshape(1, 3)
                    if (nameInfo[2] is "3"):
                        label = np.array(L_AD).reshape(1, 3)
                    labels = np.row_stack((labels, label))
                    if(image_count == batch_size*(batch + 1)):
                        break

                if(image_count > 260 and image_count <= 302):
                    image_count = image_count + 1
                    png_image = Image.open(path_224_train + pngFileName)
                    datas.append(np.array(png_image) / 255.0)
                    if (nameInfo[2] is "1"):
                        label = np.array(L_CN).reshape(1, 3)
                    if (nameInfo[2] is "2"):
                        label = np.array(L_MCI).reshape(1, 3)
                    if (nameInfo[2] is "3"):
                        label = np.array(L_AD).reshape(1, 3)
                    labels = np.row_stack((labels, label))


    if(target == 2):
        listPngFileName = os.listdir(path_224_test)
        for pngFileName in listPngFileName:
            fileName = pngFileName.split(".")
            nameInfo = fileName[0].split("-")
            if (nameInfo[1] == str(position)):
                png_image = Image.open(path_224_test + pngFileName)
                # 图像
                datas.append(np.array(png_image) / 255.0)
                if (nameInfo[2] is "1"):
                    label = np.array(L_CN).reshape(1, 3)
                if (nameInfo[2] is "2"):
                    label = np.array(L_MCI).reshape(1, 3)
                if (nameInfo[2] is "3"):
                    label = np.array(L_AD).reshape(1, 3)

                labels = np.row_stack((labels, label))


    # if(target == 3):
    #     listPngFileName = os.listdir(path_224)
    #     for pngFileName in listPngFileName:
    #         fileName = pngFileName.split(".")
    #         nameInfo = fileName[0].split("-")
    #         image_count = image_count + 1
    #         if (image_count > batch_size * batch and image_count <= batch_size * (batch + 1)):
    #             png_image = Image.open(path_224 + pngFileName)
    #             datas.append(np.array(png_image) / 255.0)
    #             if (nameInfo[2] is "1"):
    #                 label = np.array(L_CN).reshape(1, 3)
    #             if (nameInfo[2] is "2"):
    #                 label = np.array(L_MCI).reshape(1, 3)
    #             if (nameInfo[2] is "3"):
    #                 label = np.array(L_AD).reshape(1, 3)
    #             labels = np.row_stack((labels, label))
    #             if (image_count == batch_size * (batch + 1)):
    #                 break
    #
    #         if (image_count > 300 and image_count <= 302):
    #             image_count = image_count + 1
    #             png_image = Image.open(path_224 + pngFileName)
    #             datas.append(np.array(png_image) / 255.0)
    #             if (nameInfo[2] is "1"):
    #                 label = np.array(L_CN).reshape(1, 3)
    #             if (nameInfo[2] is "2"):
    #                 label = np.array(L_MCI).reshape(1, 3)
    #             if (nameInfo[2] is "3"):
    #                 label = np.array(L_AD).reshape(1, 3)
    #
    #             labels = np.row_stack((labels, label))

    datas = np.array(datas)
    labels = np.array(labels)

    # if(target == 1):
    #     print("Get data batch for train *** " + str(batch + 1) )
    #     #print("x=" + str(datas.shape) + " y=" + str(labels.shape))
    # if(target == 2):
    #     print("Get data batch for test")
       #print("x=" + str(datas.shape) + " y=" + str(labels.shape))
    # if(target == 3):
    #     print("Get data batch for feature")
    #     print("x=" + str(datas.shape) + " y=" + str(labels.shape))

    return datas, labels



def read_data_2(batch_size, batch, target):
    datas_list = []
    labels = np.random.rand(1, 3)
    labels = np.delete(labels, 0, axis=0)
    if(target == 1):
        for i in range(batch_size*batch, batch_size*(batch + 1)):
            datas_list.append(fc8_merge_x_train[i])
            labels = np.row_stack((labels, fc8_merge_y_train[i,:]))
        datas = np.array(datas_list)
        datas = np.reshape(datas, (-1, 3000))
        #datas = preprocessing.scale(datas)
        datas = preprocessing.MinMaxScaler().fit_transform(datas)
    if(target == 2):
        #datas = fc8_merge_x_test
        datas = np.array(fc8_merge_x_test)
        datas = np.reshape(datas, (-1, 3000))
        #datas = preprocessing.scale(datas)
        datas = preprocessing.MinMaxScaler().fit_transform(datas)
        labels = np.row_stack((labels, fc8_merge_y_test))

    return datas, labels


def read_data_3(batch_size, batch, target):
    datas_list = []
    labels = np.random.rand(1, 3)
    labels = np.delete(labels, 0, axis=0)
    if (target == 1):
        for i in range(batch_size * batch, batch_size * (batch + 1)):
            datas_list.append(fc8_merge_x_train[i])
            labels = np.row_stack((labels, fc8_merge_y_train[i, :]))
        datas = np.array(datas_list)
        datas = np.reshape(datas, (-1, 1000))
        # datas = preprocessing.scale(datas)
        datas = preprocessing.MinMaxScaler().fit_transform(datas)
    if (target == 2):
        # datas = fc8_merge_x_test
        datas = np.array(fc8_merge_x_test)
        datas = np.reshape(datas, (-1, 1000))
        # datas = preprocessing.scale(datas)
        datas = preprocessing.MinMaxScaler().fit_transform(datas)
        labels = np.row_stack((labels, fc8_merge_y_test))

    return datas, labels


def conv_op(input_op, name, kh, kw, n_out, dh, dw, p):
    # 获取输入数据的通道数
    n_in = input_op.get_shape()[-1].value

    with tf.name_scope(name) as scope:
        # 创建卷积核，shape的值的意义参见alexNet
        kernel = tf.get_variable(scope + "w", shape=[kh, kw, n_in, n_out], dtype=tf.float32,
                                 initializer=tf.contrib.layers.xavier_initializer_conv2d())
        # 卷积操作
        conv = tf.nn.conv2d(input_op, kernel, (1, dh, dw, 1), padding='SAME')
        # 初始化bias为0
        bias_init_val = tf.constant(0.0, shape=[n_out], dtype=tf.float32)
        biases = tf.Variable(bias_init_val, trainable=True, name='b')
        # 将卷积后结果与biases加起来
        z = tf.nn.bias_add(conv, biases)
        # 使用激活函数relu进行非线性处理
        activation = tf.nn.relu(z, name=scope)
        # 将卷积核和biases加入到参数列表
        p += [kernel, biases]
        # tf.image.resize_images()
        # 卷积层输出作为函数结果返回
        return activation


'''全连接层FC创建函数'''

def fc_op(input_op, name, n_out, p):
    # 获取input_op的通道数
    n_in = input_op.get_shape()[-1].value

    with tf.name_scope(name) as scope:
        # 初始化全连接层权重
        kernel = tf.get_variable(scope + "w", shape=[n_in, n_out], dtype=tf.float32,
                                 initializer=tf.contrib.layers.xavier_initializer())
        # 初始化biases为0.1而不为0，避免dead neuron
        biases = tf.Variable(tf.constant(0.1, shape=[n_out], dtype=tf.float32), name='b')
        # Computes Relu(x * weight + biases)
        activation = tf.nn.relu_layer(input_op, kernel, biases, name=scope)
        # 将权重和biases加入到参数列表
        p += [kernel, biases]
        # activation作为函数结果返回
        return activation


'''最大池化层创建函数'''

def mpool_op(input_op, name, kh, kw, dh, dw):
    return tf.nn.max_pool(input_op,
                          ksize=[1, kh, kw, 1],  # 池化窗口大小
                          strides=[1, dh, dw, 1],  # 池化步长
                          padding='SAME',
                          name=name)


'''创建VGGNet-16-D的网络结构
input_op为输入数据，keep_prob为控制dropoout比率的一个placeholder
'''


def vggnet_16(position):

    x = tf.placeholder(tf.float32, [None, 224, 224 ,3])
    y = tf.placeholder(tf.float32, [None, 3])
    keep_prob = tf.placeholder(tf.float32)

    p = []

    '''D-第一段'''
    # 第一段卷积网络第一个卷积层，输出尺寸224*224*64，卷积后通道数(厚度)由3变为64
    conv1_1 = conv_op(x, name="conv1_1", kh=3, kw=3, n_out=64, dh=1, dw=1, p=p)
    # 第一段卷积网络第二个卷积层，输出尺寸224*224*64
    conv1_2 = conv_op(conv1_1, name="conv1_2", kh=3, kw=3, n_out=64, dh=1, dw=1, p=p)
    # 第一段卷积网络的最大池化层，经过池化后输出尺寸变为112*112*64
    pool1 = mpool_op(conv1_2, name="pool1", kh=2, kw=2, dw=2, dh=2)

    '''D-第二段'''
    # 第二段卷积网络第一个卷积层，输出尺寸112*112*128，卷积后通道数由64变为128
    conv2_1 = conv_op(pool1, name="conv2_1", kh=3, kw=3, n_out=128, dh=1, dw=1, p=p)
    # 第二段卷积网络第二个卷积层，输出尺寸112*112*128
    conv2_2 = conv_op(conv2_1, name="conv2_1", kh=3, kw=3, n_out=128, dh=1, dw=1, p=p)
    # 第二段卷积网络的最大池化层，经过池化后输出尺寸变为56*56*128
    pool2 = mpool_op(conv2_2, name="pool2", kh=2, kw=2, dw=2, dh=2)

    '''D-第三段'''
    # 第三段卷积网络第一个卷积层，输出尺寸为56*56*256，卷积后通道数由128变为256
    conv3_1 = conv_op(pool2, name="conv3_1", kh=3, kw=3, n_out=256, dh=1, dw=1, p=p)
    # 第三段卷积网络第二个卷积层，输出尺寸为56*56*256
    conv3_2 = conv_op(conv3_1, name="conv3_2", kh=3, kw=3, n_out=256, dh=1, dw=1, p=p)
    # 第三段卷积网络第三个卷积层，输出尺寸为56*56*256
    conv3_3 = conv_op(conv3_2, name="conv3_3", kh=3, kw=3, n_out=256, dh=1, dw=1, p=p)
    # 第三段卷积网络的最大池化层，池化后输出尺寸变为28*28*256
    pool3 = mpool_op(conv3_3, name="pool3", kh=2, kw=2, dh=2, dw=2)

    '''D-第四段'''
    # 第四段卷积网络第一个卷积层，输出尺寸为28*28*512，卷积后通道数由256变为512
    conv4_1 = conv_op(pool3, name="conv4_1", kh=3, kw=3, n_out=512, dh=1, dw=1, p=p)
    # 第四段卷积网络第二个卷积层，输出尺寸为28*28*512
    conv4_2 = conv_op(conv4_1, name="conv4_2", kh=3, kw=3, n_out=512, dh=1, dw=1, p=p)
    # 第四段卷积网络第三个卷积层，输出尺寸为28*28*512
    conv4_3 = conv_op(conv4_2, name="conv4_3", kh=3, kw=3, n_out=512, dh=1, dw=1, p=p)
    # 第四段卷积网络的最大池化层，池化后输出尺寸为14*14*512
    pool4 = mpool_op(conv4_3, name="pool4", kh=2, kw=2, dh=2, dw=2)

    '''D-第五段'''
    # 第五段卷积网络第一个卷积层，输出尺寸为14*14*512
    conv5_1 = conv_op(pool4, name="conv5_1", kh=3, kw=3, n_out=512, dh=1, dw=1, p=p)
    # 第五段卷积网络第二个卷积层，输出尺寸为14*14*512
    conv5_2 = conv_op(conv5_1, name="conv5_2", kh=3, kw=3, n_out=512, dh=1, dw=1, p=p)
    # 第五段卷积网络第三个卷积层，输出尺寸为14*14*512
    conv5_3 = conv_op(conv5_2, name="conv5_3", kh=3, kw=3, n_out=512, dh=1, dw=1, p=p)
    # 第五段卷积网络的最大池化层，池化后尺寸为7*7*512
    pool5 = mpool_op(conv5_3, name="conv5_3", kh=2, kw=2, dh=2, dw=2)

    '''对卷积网络的输出结果进行扁平化，将每个样本化为一个长度为25088的一维向量'''
    shp = pool5.get_shape()
    flattened_shape = shp[1].value * shp[2].value * shp[3].value  # 图像的长、宽、厚度相乘，即7*7*512=25088
    # -1表示该样本有多少个是自动计算得出的，得到一个矩阵，准备传入全连接层
    resh1 = tf.reshape(pool5, [-1, flattened_shape], name="resh1")

    '''全连接层,共三个'''
    fc6 = fc_op(resh1, name="fc6", n_out=4096, p=p)
    fc6_drop = tf.nn.dropout(fc6, 0.5, name="fc6_drop")  # dropout层，keep_prob数据待外部传入

    fc7 = fc_op(fc6_drop, name="fc7", n_out=4096, p=p)
    fc7_drop = tf.nn.dropout(fc7, 0.5, name="fc7_drop")

    fc8 = fc_op(fc7_drop, name="fc7", n_out=1000, p=p)
    fc8_drop = tf.nn.dropout(fc7, 0.5, name="fc8_drop")

    # 最后一个全连接层
    fc9 = fc_op(fc8_drop, name="fc8", n_out=3, p=p)
    prediction = tf.nn.softmax(fc9)  # 使用softmax进行处理得到分类输出概率



    #交叉代价函数
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y,logits=prediction))

    #使用AdamOptimizer进行优化
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)

    #结果存放在一个布尔列表中
    correct_prediction = tf.equal(tf.argmax(prediction,1),tf.argmax(y,1)) #argmax返回一维张量中最大的值所在的位置

    #求准确率
    accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

    global max_epoch
    global max_batch


    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        print("************************** " +  str(position) + " **************************")
        ''' -------------------------------------------------------- '''
        acc_max = 0

        ''' -------------------------------------------------------- '''
        for epoch in range(1):
            #print("Start Train Epoch " + str(epoch + 1) + " **************************")
            for batch in range(10):
                train_x, train_y = read_data_1(position, 13, batch, 1)
                sess.run(train_step,feed_dict={x:train_x,y:train_y ,keep_prob:0.5})
                print("@@@@@@ ")
                test_x, test_y = read_data_1(position, None, None, 2)
                acc = sess.run(accuracy,feed_dict={x:test_x,y:test_y ,keep_prob:1.0})
                print("Vggnet Epoch" + str(epoch + 1) + ",Batch" + str(batch + 1) + ",Testing Accuracy=" + str(acc))
                if (acc > acc_max):
                    saver = tf.train.Saver()
                    saver.save(sess,path_model + "vggnet_16_nn.ckpt")
                    acc_max = acc
                    max_epoch = epoch
                    max_batch = batch


        # train_x, train_y = read_data_1(1, 1, 1, 1)
        # pool5 = sess.run(pool5, feed_dict={x: train_x})
        # pool5_transpose = sess.run(tf.transpose(pool5, [3, 0, 1, 2]))
        # fig6, ax6 = plt.subplots(nrows=1, ncols=8, figsize=(8, 1))
        # for i in range(8):
        #     ax6[i].imshow(pool5_transpose[i][0])
        #     plt.show()

        print("Position" + str(position) + ",Max Accuracy=" + str(acc_max) + ",Epoch=" + str(max_epoch) + ",Batch=" + str(max_batch) + "\n")

        print("Position " + str(position) + " Start epoch feature extraction")
        # 特征提取，输出fc8的特征向量

        ''' --------------------------------------------------------  '''
        fc8_output_list = []
        fc8_labels_array = []
        fc8_labels_array = np.random.rand(1, 3)
        fc8_labels_array = np.delete(fc8_labels_array, 0, axis=0)

        global fc8_temp_1_x
        global fc8_temp_1_y
        global fc8_temp_2_x
        global fc8_temp_2_y
        global fc8_temp_3_x
        global fc8_temp_3_y

        ''' --------------------------------------------------------  '''
        saver.restore(sess,path_model + "vggnet_16_nn.ckpt")
        listPngFileName = os.listdir(path_224)
        for pngFileName in listPngFileName:
            image_data = []
            fileName = pngFileName.split(".")
            nameInfo = fileName[0].split("-")
            if (nameInfo[1] == str(position)):
                png_image = Image.open(path_224 + pngFileName)
                image_data.append(np.array(png_image) / 255.0)
                fc8_output_list.append(sess.run(fc8, feed_dict={x:image_data}))
                if (nameInfo[2] is "1"):
                    label = np.array(L_CN).reshape(1, 3)
                if (nameInfo[2] is "2"):
                    label = np.array(L_MCI).reshape(1, 3)
                if (nameInfo[2] is "3"):
                    label = np.array(L_AD).reshape(1, 3)
                fc8_labels_array = np.row_stack((fc8_labels_array, label))

        if (position == 1):
            fc8_temp_1_x = fc8_output_list
            fc8_temp_1_y = fc8_labels_array
        if (position == 2):
            fc8_temp_2_x = fc8_output_list
            fc8_temp_2_y = fc8_labels_array
        if (position == 3):
            fc8_temp_3_x = fc8_output_list
            fc8_temp_3_y = fc8_labels_array

        # if(position == 1):
        #     #fc8_temp_1_x = fc8_output_list
        #     # fc8_output_list = json.dumps(fc8_output_list)
        #     # fc8_1_x = open(r"D:\cjh-model\AD-Model\AD-ADNI-DATA\clinical_data\data_temp\fc8_1_x.txt","w",encoding="UTF-8")
        #     # fc8_1_x.write(fc8_output_list)
        #     # np.savetxt(temp_path + "fc8_" + str(position) + "_y.txt", fc8_labels_array,fmt='%s')
        # if (position == 2):
        #     # fc8_output_list = json.dumps(fc8_output_list)
        #     # fc8_2_x = open(r"D:\cjh-model\AD-Model\AD-ADNI-DATA\clinical_data\data_temp\fc8_2_x.txt", "w",encoding="UTF-8")
        #     # fc8_2_x.write(fc8_output_list)
        #     # np.savetxt(temp_path + "fc8_" + str(position) + "_y.txt", fc8_labels_array, fmt='%s')
        # if (position == 3):
        #     # fc8_output_list = json.dumps(fc8_output_list)
        #     # fc8_3_x = open(r"D:\cjh-model\AD-Model\AD-ADNI-DATA\clinical_data\data_temp\fc8_3_x.txt", "w",encoding="UTF-8")
        #     # fc8_3_x.write(fc8_output_list)
        #     # np.savetxt(temp_path + "fc8_" + str(position) + "_y.txt", fc8_labels_array, fmt='%s')

    sess.close()

            # print(fc8_output_array.shape)
            # print(fc8_labels_array.shape)
            # print(fc8_labels_array)
            # np.savetxt(temp_path + "fc8_" + str(position) + "_x.txt", fc8_output_array,fmt='%s')
            # np.savetxt(temp_path + "fc8_" + str(position) + "_y.txt", fc8_labels_array,fmt='%s')

        # # 特征提取，输出fc8的特征向量
        # fc8_output_list = []
        # fc8_labels = []
        # for batch in range(10):
        #     all_x, all_y = read_data(position, 30, batch, 3)
        #     fc8_output_list.append(sess.run(fc8, feed_dict={x: all_x}))
        #     fc8_labels.append(all_y)
        # fc8_output_array = np.array(fc8_output_list)
        # fc8_labels_array = np.array(fc8_labels)
        # print("fc8_output ****************************")
        # print(fc8_output_array.shape)
        # print(fc8_labels_array.shape)
        # print(temp_path + "fc8_" + str(position) + "_x.txt")
        # # np.savetxt(temp_path + "fc8_" + str(position) + "_x.txt", fc8_output_array,fmt='%s')
        # # np.savetxt(temp_path + "fc8_" + str(position) + "_y.txt", fc8_labels_array,fmt='%s')

def nn():
    # 定义两个placeholder
    x = tf.placeholder(tf.float32, [None, 3000])
    y = tf.placeholder(tf.float32, [None, 3])
    keep_prob = tf.placeholder(tf.float32)

    input = tf.reshape(x,[-1,3000])
    # 创建一个简单的神经网络

    '''定义隐层一'''
    # w_1 = tf.Variable(tf.truncated_normal([3000, 500]))
    # l_1 = tf.matmul(input, w_1)
    # l1_drop = tf.nn.dropout(l_1, keep_prob)

    '''定义隐层二'''
    # w_2 = tf.Variable(tf.truncated_normal([2000, 1000], mean=0.0, stddev=1.0, dtype=tf.float32, seed=None, name=None))
    # l_2 = tf.matmul(l1_drop, w_2)
    # l2_drop = tf.nn.dropout(l_2, keep_prob)

    w_3 = tf.Variable(tf.truncated_normal([3000, 3]))

    # l_t = tf.matmul(input , w_3)
    # lt_drop = tf.nn.dropout(l_t, keep_prob)
    # prediction = tf.nn.softmax(lt_drop)

    prediction = tf.nn.softmax(tf.matmul(input, w_3))

    # 使用交叉熵函数
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=prediction))
    # 使用梯度下降算法
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss)

    # 初始化变量
    init = tf.global_variables_initializer()

    # 结果存放在一个布尔型列表中
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(prediction, 1))  # argmax返回一维张量中最大值所在的位置

    # 求准确率
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))  # cast将布尔型转换为浮点型

    #saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(init)
        # sess.run(train_step, feed_dict={x: batch_xs, y: batch_ys})

        ''' -------------------------------------------------------- '''
        for epoch in range(1):
            # print("Start Train Epoch " + str(epoch + 1) + " **************************")
            for batch in range(1):
                train_x, train_y = read_data_2(10, batch, 1)
                sess.run(train_step, feed_dict={x: train_x, y: train_y, keep_prob: 0.5})

                test_x, test_y = read_data_2(None, None, 2)
                acc = sess.run(accuracy, feed_dict={x:  test_x, y: test_y, keep_prob: 1.0})
                print("NN Epoch" + str(epoch + 1) + ",Batch" + str(batch + 1) + ",Testing Accuracy=" + str(acc))

# def nn_1():
#     # 定义两个placeholder
#     x = tf.placeholder(tf.float32, [None, 1000])
#     y = tf.placeholder(tf.float32, [None, 3])
#     keep_prob = tf.placeholder(tf.float32)
#
#     input = tf.reshape(x,[-1,1000])
#     # 创建一个简单的神经网络
#
#     # '''定义隐层一'''
#     # w_1 = tf.Variable(tf.truncated_normal([3000, 2000]))
#     # l_1 = tf.matmul(input, w_1)
#     # l1_drop = tf.nn.dropout(l_1, keep_prob)
#     #
#     # '''定义隐层二'''
#     # # w_2 = tf.Variable(tf.truncated_normal([2000, 1000], mean=0.0, stddev=1.0, dtype=tf.float32, seed=None, name=None))
#     # # l_2 = tf.matmul(l1_drop, w_2)
#     # # l2_drop = tf.nn.dropout(l_2, keep_prob)
#
#     w_3 = tf.Variable(tf.truncated_normal([1000, 3]))
#
#     # l_t = tf.matmul(input , w_3)
#     # lt_drop = tf.nn.dropout(l_t, keep_prob)
#     # prediction = tf.nn.softmax(lt_drop)
#
#     prediction = tf.nn.softmax(tf.matmul(input , w_3))
#     print(prediction)
#     # 使用交叉熵函数
#     loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=prediction))
#     # 使用梯度下降算法
#     train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss)
#
#     # 初始化变量
#     init = tf.global_variables_initializer()
#
#     # 结果存放在一个布尔型列表中
#     correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(prediction, 1))  # argmax返回一维张量中最大值所在的位置
#
#     # 求准确率
#     accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))  # cast将布尔型转换为浮点型
#
#     #saver = tf.train.Saver()
#
#     with tf.Session() as sess:
#         sess.run(init)
#         # sess.run(train_step, feed_dict={x: batch_xs, y: batch_ys})
#
#         ''' -------------------------------------------------------- '''
#         for epoch in range(100):
#             # print("Start Train Epoch " + str(epoch + 1) + " **************************")
#             for batch in range(52):
#                 train_x, train_y = read_data_3(5, batch, 1)
#                 print(train_x.shape)
#                 sess.run(train_step, feed_dict={x: train_x, y: train_y, keep_prob: 0.03})
#
#                 test_x, test_y = read_data_3(None, None, 2)
#                 acc = sess.run(accuracy, feed_dict={x:  test_x, y: test_y, keep_prob: 1.0})
#                 print("NN Epoch" + str(epoch + 1) + ",Batch" + str(batch + 1) + ",Testing Accuracy=" + str(acc))





if __name__ == "__main__":


    gpu_config()
    # read_data(1)
    # divide_data(260)
    vggnet_16(1)

    vggnet_16(2)

    vggnet_16(3)

    '''
    fc8_merge_x     list     (302, 3)
    fc8_merge_y     array    (1, 3000)
    '''


    divide_data_2(260)
    nn()

    # divide_data_3(260)
    # nn_1()


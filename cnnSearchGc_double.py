import matplotlib.pyplot as plt
import random
import numpy as np
import tensorflow as tf
from astropy.io import fits
import os
import cv2
import sys
import seaborn as sn
import pandas as pd
from tensorflow.examples.tutorials.mnist import input_data
from sklearn import metrics
from from_literature.utils_me import img_show, delete_all, read_text


def load_imgs_single_fit():
    img_name = ['positive_2channels_augment.fits', 'negative_2channels.fits']
    label_name = [[0, 1], [1, 0]]
    imgs = []
    labels = []
    fit_path = 'D:/国家天文台/PAPER2/双通道/训练样本/'
    fit_path = 'D:/国家天文台/PAPER2/双通道用回自己的图片预处理/训练样本/'
    for i in np.arange(0, 2):
        singles = fits.open(fit_path + img_name[i])[0].data
        # if i == 1:
        #     singles = fits.open(fit_path + img_name[i])[0].data
        for single in singles:
            imgs.append(single)
            labels.append(np.array(label_name[i]))
    indexs = [x for x in range(len(imgs))]
    random.shuffle(indexs)
    imgs = np.array(imgs)[indexs]
    labels = np.array(labels)[indexs]
    print(len(imgs), len(labels))
    return [imgs, labels]


def train():
    x_y = load_imgs_single_fit()
    length = len(x_y[0])

    input_x = tf.placeholder(tf.float32, [None, 2 * 56 * 56])

    output_y = tf.placeholder(tf.int32, [None, 2])

    input_x_images = tf.reshape(input_x, [-1, 56, 56, 2])

    ith1 = 400
    ith2 = 50
    batch = 50

    test_x = x_y[0][length - ith1:length - ith2]  # image
    test_y = x_y[1][length - ith1:length - ith2]  # label

    conv0 = tf.layers.conv2d(
        inputs=input_x_images,
        filters=64,
        kernel_size=[5, 5],
        strides=1,
        padding='same',
        activation=tf.nn.relu
    )

    pool0 = tf.layers.max_pooling2d(
        inputs=conv0,
        pool_size=[2, 2],
        strides=2
    )
    print(pool0)

    conv1 = tf.layers.conv2d(
        inputs=pool0,
        filters=64,
        kernel_size=[5, 5],
        strides=1,
        padding='same',
        activation=tf.nn.relu
    )

    pool1 = tf.layers.max_pooling2d(
        inputs=conv1,
        pool_size=[2, 2],
        strides=2
    )
    print(pool1)

    conv2 = tf.layers.conv2d(
        inputs=pool1,
        filters=128,
        kernel_size=[5, 5],
        strides=1,
        padding='same',
        activation=tf.nn.relu
    )

    pool2 = tf.layers.max_pooling2d(
        inputs=conv2,
        pool_size=[2, 2],
        strides=2
    )

    print(pool2)
    flat = tf.reshape(pool2, [-1, 7 * 7 * 128])

    dense = tf.layers.dense(
        inputs=flat,
        units=1024,
        activation=tf.nn.relu
    )

    print(dense)
    dropout = tf.layers.dropout(
        inputs=dense,
        rate=0.5,
    )
    print(dropout)

    logits = tf.layers.dense(
        inputs=dropout,
        units=2
    )

    print(logits)

    loss = tf.losses.softmax_cross_entropy(onehot_labels=output_y,
                                           logits=logits)

    print(loss)
    train_op = tf.train.GradientDescentOptimizer(learning_rate=0.001).minimize(loss)

    accuracy_op = tf.metrics.accuracy(
        labels=tf.argmax(output_y, axis=1),
        predictions=tf.argmax(logits, axis=1)
    )[1]

    sess = tf.Session()
    init = tf.group(tf.global_variables_initializer(),
                    tf.local_variables_initializer())
    sess.run(init)

    for j in np.arange(0, 35):
        for i in np.arange(0, length - ith1, batch):
            train_loss, train_op_ = sess.run([loss, train_op],
                                             {input_x: x_y[0][i:i + batch], output_y: x_y[1][i:i + batch]})
            if i % ith1 == 0:
                test_accuracy = sess.run(accuracy_op, {input_x: test_x, output_y: test_y})
                print(j, ":Step=%d, Train loss=%.4f,[Test accuracy=%.2f]" % (i, train_loss, test_accuracy))
    test_output = sess.run(logits, {input_x: x_y[0][length - ith2:length]})
    inferenced_y = np.argmax(test_output, 1)
    print(inferenced_y, 'Inferenced numbers')
    print(np.argmax(x_y[1][length - ith2:length], 1), 'Real numbers')
    saver = tf.train.Saver(max_to_keep=1)
    # saver.save(sess, 'D:/国家天文台/PAPER2/双通道/训练模型保存/模型1/final_model.ckpt')
    saver.save(sess, 'D:/国家天文台/PAPER2/双通道用回自己的图片预处理/模型1/final_model.ckpt')
    sess.close()


def predict():  
    x_y = load_imgs_single_fit()
    length = len(x_y[0])
    input_x = tf.placeholder(tf.float32, [None, 2 * 56 * 56])
    output_y = tf.placeholder(tf.int32, [None, 2])
    input_x_images = tf.reshape(input_x, [-1, 56, 56, 2])

    test_x = x_y[0][length - 400:length - 0]  # image
    test_y = x_y[1][length - 400:length - 0]  # label

    conv0 = tf.layers.conv2d(
        inputs=input_x_images,
        filters=64,
        kernel_size=[5, 5],
        strides=1,
        padding='same',
        activation=tf.nn.relu
    )

    pool0 = tf.layers.max_pooling2d(
        inputs=conv0,
        pool_size=[2, 2],
        strides=2
    )
    print(pool0)

    conv1 = tf.layers.conv2d(
        inputs=pool0,
        filters=64,
        kernel_size=[5, 5],
        strides=1,
        padding='same',
        activation=tf.nn.relu
    )

    pool1 = tf.layers.max_pooling2d(
        inputs=conv1,
        pool_size=[2, 2],
        strides=2
    )
    print(pool1)

    conv2 = tf.layers.conv2d(
        inputs=pool1,
        filters=128,
        kernel_size=[5, 5],
        strides=1,
        padding='same',
        activation=tf.nn.relu
    )

    pool2 = tf.layers.max_pooling2d(
        inputs=conv2,
        pool_size=[2, 2],
        strides=2
    )

    print(pool2)
    flat = tf.reshape(pool2, [-1, 7 * 7 * 128])
    dense = tf.layers.dense(
        inputs=flat,
        units=1024,
        activation=tf.nn.relu
    )

    print(dense)
    dropout = tf.layers.dropout(
        inputs=dense,
        rate=0.5,
    )
    print(dropout)

    logits = tf.layers.dense(
        inputs=dropout,
        units=2
    )
    loss = tf.losses.softmax_cross_entropy(onehot_labels=output_y,
                                           logits=logits)

    print(loss)
    train_op = tf.train.GradientDescentOptimizer(learning_rate=0.001).minimize(loss)

    accuracy_op = tf.metrics.accuracy(
        labels=tf.argmax(output_y, axis=1),
        predictions=tf.argmax(logits, axis=1)
    )[1]
    path = 'E:/博士/toPredictImgsDouble2/'
    fileNames = os.listdir(path)
    with tf.Session() as sess:
        init = tf.group(tf.global_variables_initializer(),
                        tf.local_variables_initializer())
        sess.run(init)
        saver = tf.train.Saver()
        saver.restore(sess, 'D:/国家天文台/PAPER2/双通道用回自己的图片预处理/模型1/final_model.ckpt')
        test_accuracy = sess.run(accuracy_op,
                                 {input_x: test_x, output_y: test_y})
        print(test_accuracy)
        for fileName in fileNames:
            if fileName.endswith('.fits'):
                coordName = 'D:/国家天文台/PAPER2/双通道/CNN预测结果/' + fileName[:-5] + '.txt'
                if not os.path.exists('D:/国家天文台/PAPER2/双通道用回自己的图片预处理/CNN预测结果/' + fileName[:-5] + '.txt'):
                    coordName = 'D:/国家天文台/PAPER2/双通道用回自己的图片预处理/CNN预测结果/' + fileName[:-5] + '.txt'
                    if os.path.exists(coordName):
                        os.remove(coordName)
                    total = open(coordName, 'w')
                    print(fileName)
                    coord = read_text(path + fileName.split('.')[0] + '.txt')
                    precast = fits.open(path + fileName)[0].data
                    precast = np.array(precast)
                    gc = []
                    for i in np.arange(0, len(precast)):
                        test_output = sess.run(logits, {input_x: precast[i:i + 1]})
                        predict_score = (np.exp(test_output[0]) / np.sum(np.exp(test_output[0])))[1]
                        if predict_score >= 0.99:
                            gc.append(precast[i])
                            for sub_coord in coord[i]:
                                total.writelines(sub_coord + ' ')
                            total.writelines(str(predict_score)+'\n')
                    total.close()
                    print(len(gc), len(read_text(coordName)))
                    if len(gc) > 0:
                        grey = fits.PrimaryHDU(gc)
                        grey_hdu = fits.HDUList([grey])
                        file = 'D:/国家天文台/PAPER2/双通道用回自己的图片预处理/CNN预测结果/' + fileName
                        if os.path.exists(file):
                            os.remove(file)
                        grey_hdu.writeto(file)
                        grey_hdu.close()
                    else:
                        os.remove(coordName)
        sess.close()


def predict2():  # 测试predict单个fits,
    x_y = load_imgs_single_fit()
    length = len(x_y[0])
    input_x = tf.placeholder(tf.float32, [None, 2 * 56 * 56])
    output_y = tf.placeholder(tf.int32, [None, 2])
    input_x_images = tf.reshape(input_x, [-1, 56, 56, 2])

    test_x = x_y[0][length - 400:length - 0]  # image
    test_y = x_y[1][length - 400:length - 0]  # label

    conv0 = tf.layers.conv2d(
        inputs=input_x_images,
        filters=64,
        kernel_size=[5, 5],
        strides=1,
        padding='same',
        activation=tf.nn.relu
    )

    pool0 = tf.layers.max_pooling2d(
        inputs=conv0,
        pool_size=[2, 2],
        strides=2
    )
    print(pool0)

    conv1 = tf.layers.conv2d(
        inputs=pool0,
        filters=64,
        kernel_size=[5, 5],
        strides=1,
        padding='same',
        activation=tf.nn.relu
    )

    pool1 = tf.layers.max_pooling2d(
        inputs=conv1,
        pool_size=[2, 2],
        strides=2
    )
    print(pool1)

    conv2 = tf.layers.conv2d(
        inputs=pool1,
        filters=128,
        kernel_size=[5, 5],
        strides=1,
        padding='same',
        activation=tf.nn.relu
    )

    pool2 = tf.layers.max_pooling2d(
        inputs=conv2,
        pool_size=[2, 2],
        strides=2
    )

    print(pool2)
    flat = tf.reshape(pool2, [-1, 7 * 7 * 128])
    dense = tf.layers.dense(
        inputs=flat,
        units=1024,
        activation=tf.nn.relu
    )

    print(dense)
    dropout = tf.layers.dropout(
        inputs=dense,
        rate=0.5,
    )
    print(dropout)

    logits = tf.layers.dense(
        inputs=dropout,
        units=2
    )
    loss = tf.losses.softmax_cross_entropy(onehot_labels=output_y,
                                           logits=logits)

    print(loss)
    train_op = tf.train.GradientDescentOptimizer(learning_rate=0.001).minimize(loss)

    accuracy_op = tf.metrics.accuracy(
        labels=tf.argmax(output_y, axis=1),
        predictions=tf.argmax(logits, axis=1)
    )[1]
    with tf.Session() as sess:
        init = tf.group(tf.global_variables_initializer(),
                        tf.local_variables_initializer())
        sess.run(init)
        saver = tf.train.Saver()
        saver.restore(sess, 'D:/国家天文台/PAPER2/双通道/训练模型保存/模型1/final_model.ckpt')
        test_accuracy = sess.run(accuracy_op,
                                 {input_x: test_x, output_y: test_y})
        print(test_accuracy)

        precast = fits.open('D:/国家天文台/PAPER2/双通道/训练样本/positive_2channels_augment.fits')[0].data
        precast = np.array(precast)
        gc = []
        for i in np.arange(0, len(precast)):
            test_output = sess.run(logits, {input_x: precast[i:i + 1]})
            if (np.exp(test_output[0]) / np.sum(np.exp(test_output[0])))[1] >= 0.9:
                gc.append(precast[i])
        print(len(precast), len(gc))
        sess.close()


def multi_channels():  # 多通道图像处理
    from skimage import io
    # path = 'D:/国家天文台/PAPER2/PAndAS图像/cutSizeCompare/0.jpg'
    # path = 'D:/国家天文台/PAPER2/测试/test2.jpg'
    path = 'D:/国家天文台/PAPER2/测试/test.jpg'
    img = io.imread(path)
    (b, g, r) = cv2.split(img)
    print(b[0][1], g[0][1], r[0][1])
    print(np.array(img).shape)
    print(img[0][1])
    img2 = np.dstack((b, g))  # 多通道合并
    print(img2[0][1])
    plt.imshow(g, cmap='gray')
    plt.show()


def merge_decompose_channels():  # 拆分、并合多通道
    path2 = 'E:/博士/listdir/m001_i.fit/'
    path = 'D:/国家天文台/PAPER2/PAndAS图像/g_band/m026_g.fit/'
    pos = read_text('D:/国家天文台/PAPER2/预测结果/目视检查_56_56/最终候选体的完整catalogue.txt')
    cut_size = 28
    for single in pos:
        fit_name = path + 'm' + single[-1].zfill(3) + '_g.fit'
        fit = fits.open(fit_name)
        fit_name2 = path2 + 'm' + single[-1].zfill(3) + '_i.fit'
        fit2 = fits.open(fit_name2)
        img = fit[int(single[2])].data[int(float(single[4])) - cut_size:int(float(single[4])) + cut_size,
              int(float(single[3])) - cut_size:int(float(single[3])) + cut_size]
        img2 = fit2[int(single[2])].data[int(float(single[9])) - cut_size:int(float(single[9])) + cut_size,
               int(float(single[8])) - cut_size:int(float(single[8])) + cut_size]
        if img.shape == (cut_size * 2, cut_size * 2) and img2.shape == (cut_size * 2, cut_size * 2):
            h = np.max(img)
            l = np.min(img)
            img -= l
            img /= (h-l)
            h = np.max(img2)
            l = np.min(img2)
            img2 -= l
            img2 /= (h - l)
            plt.subplot(131)
            plt.title('$g$')
            plt.imshow(img, cmap='gray')
            plt.subplot(132)
            plt.title('$i$')
            plt.imshow(img2, cmap='gray')
            plt.subplot(133)
            plt.title('$merge$')
            merge = np.dstack((img, img2))
            merge = np.reshape(merge, (-1))
            merge = np.reshape(merge, (56, 56, 2))
            g, i = cv2.split(merge)
            plt.imshow(i, cmap='gray')
            name_coord = single[0] + single[1]
            plt.suptitle(name_coord, y=0.85)
            plt.show()
            exit()
            file = 'D:/国家天文台/PAPER2/预测结果/目视检查_56_56/final_candidate_vs_g_band/' + name_coord.replace(':', '.') + '.jpg'
            plt.savefig(file)
            plt.close('all')


train()

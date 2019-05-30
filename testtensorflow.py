# dealing with the converting from scienyifc form

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from cffi.backend_ctypes import xrange
from pandas import Series
from sklearn.feature_extraction.text import TfidfVectorizer
import os
from sklearn.cross_decomposition import tests
from sklearn.linear_model import LogisticRegression
from sklearn import metrics, preprocessing
import urllib.parse
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler


class Traffic:

    def __init__(self, http, Agent, Pragma, Cache, Accept, Encoding, Charset, Language, Host, Cookie, Connection,
                 Target):
        self.http = http
        self.Agent = Agent
        self.Pragma = Pragma
        self.Cache = Cache
        self.Accept = Accept
        self.Encoding = Encoding
        self.Charset = Charset
        self.Language = Language
        self.Host = Host
        self.Cookie = Cookie
        self.Connection = Connection
        self.Target = Target


def scaleBetween(unscaledNum):
    minAllowed = 1
    maxAllowed = 100
    min = 1
    max = 100
    new = (maxAllowed - minAllowed) * (unscaledNum - min) / (max - min) + minAllowed
    return new


def ToASCII(s):
    s = str(s)
    l1 = [c for c in s]  # in Python, a string is just a sequence, so we can iterate over it!
    l2 = [ord(c) for c in s]
    s1 = ''
    for c in l2:
        s1 += str(c)
    # for i in xrange(len(s)):
    #    s1 += ord(s[i]) * 2 ** (8 * (len(s) - i - 1))
    return s1


def ToString(s):
    s1 = 0

    for i in xrange(len(s)):
        s1 += chr(s[i]) * 2 ** (8 * (len(s) - i - 1))
    return s1


def ConvertCsvAscii(dataframe):
    TrafficList = []
    tmp = []
    Target = []

    for i, row in dataframe.iterrows():
        trafic = Traffic(row["http"], row["Agent"], row["Pragma"], row["Cache"], row["Accept"], row["Encoding"],
                         row["Charset"], row["Language"], row["Host"], row["Cookie"], row["Connection"], row["Target"])
        trafic.http = str(ToASCII(trafic.http))
        trafic.Agent = str(ToASCII(trafic.Agent))
        trafic.Pragma = str(ToASCII(trafic.Pragma))
        trafic.Cache = str(ToASCII(trafic.Cache))
        trafic.Accept = str(ToASCII(trafic.Accept))
        trafic.Encoding = str(ToASCII(trafic.Encoding))
        trafic.Charset = str(ToASCII(trafic.Charset))
        trafic.Language = str(ToASCII(trafic.Language))
        trafic.Host = str(ToASCII(trafic.Host))
        trafic.Cookie = str(ToASCII(trafic.Cookie))
        trafic.Connection = str(ToASCII(trafic.Connection))
        target = str(ToASCII(str(trafic.Target)))
        trafic.Target = chr(int(target))
        TrafficList.append(trafic)

    df = pd.DataFrame([t.__dict__ for t in TrafficList])

    return df


def deleteinfvalue(d):
    TrafficList = []

    for i, row in d.iterrows():
        trafic = Traffic(row["http"], row["Agent"], row["Pragma"], row["Cache"], row["Accept"], row["Encoding"],
                         row["Charset"], row["Language"], row["Host"], row["Cookie"], row["Connection"], row["Target"])

        if trafic.http == np.inf or trafic.Agent == np.inf or trafic.Pragma == np.inf or trafic.Cache == np.inf or trafic.Accept == np.inf or trafic.Encoding == np.inf or trafic.Charset == np.inf or trafic.Language == np.inf or trafic.Host == np.inf or trafic.Cookie == np.inf or trafic.Connection == np.inf:
            trafic.http = 0
            trafic.Agent = 0
            trafic.Pragma = 0
            trafic.Cache = 0
            trafic.Accept = 0
            trafic.Encoding = 0
            trafic.Charset = 0
            trafic.Language = 0
            trafic.Host = 0
            trafic.Cookie = 0
            trafic.Connection = 0
            trafic.Target = 0
        else:
            TrafficList.append(trafic)

    dataframe = pd.DataFrame([t.__dict__ for t in TrafficList])

    return dataframe


if __name__ == "__main__":
    # for the winwods users you should change this path
    LOGDIR = "/tmp/mnist_tutorial/"

    df = pd.read_csv('Data/demo2/AllTrafficSmallASCII.csv')
    df = deleteinfvalue(df)

    scaler = MinMaxScaler()
    df[['http', 'Agent', 'Pragma', 'Cache', 'Accept', 'Encoding', 'Charset', 'Language', 'Host', 'Cookie','Connection']]= scaler.fit_transform(df[['http', 'Agent', 'Pragma', 'Cache', 'Accept', 'Encoding', 'Charset', 'Language', 'Host', 'Cookie','Connection']])
    print("Data is ready  , the training will start after a while")
    print(df)
    df =pd.DataFrame.from_records(df)

    inputX = df.loc[:,
             ['http', 'Agent', 'Pragma', 'Cache', 'Accept', 'Encoding', 'Charset', 'Language', 'Host', 'Cookie',
              'Connection']].values
    inputY = df.loc[:, ["Target"]].values

    X_train, X_test, y_train, y_test = train_test_split(inputX, inputY, test_size=0.2,
                                                        random_state=42)  # splitting data

    # Parameters
    n_input = 11
    n_hidden1 = 6
    n_hidden2 = 6
    n_output = 1
    learning_rate = 000.1
    training_epochs = 1000000
    display_step = 10000
    BATCH_SIZE = 100
    data_size = df.shape[0]
    train_size = X_train.shape[0]
    test_size = X_test.shape[0]

    tf.reset_default_graph()
    sess = tf.Session()

    X = tf.placeholder(tf.float32, name="X")
    tf.summary.histogram("inputs ", X)

    Y = tf.placeholder(tf.float32, name="output")
    tf.summary.histogram("outputs ", Y)

    with tf.name_scope("Hidden_Layer_1"):
        W1 = tf.Variable(tf.random_normal([n_input, n_hidden1]), name="W1")
        tf.summary.histogram("Weights 1", W1)

        b1 = tf.Variable(tf.random_normal([n_hidden1]), name="B1")
        tf.summary.histogram("Biases 1", b1)

        L1 = tf.nn.relu(tf.matmul(X, W1) + b1)
        tf.summary.histogram("Activation", L1)

    with tf.name_scope("Hidden_Layer_2"):
        W2 = tf.Variable(tf.random_normal([n_hidden2, n_hidden2]), name="W2")
        tf.summary.histogram("Weights 2", W2)

        b2 = tf.Variable(tf.random_normal([n_hidden2]), name="B2")
        tf.summary.histogram("Biases 2", b2)

        L2 = tf.nn.relu(tf.matmul(L1, W2) + b2)
        tf.summary.histogram("Activation", L2)

    with tf.name_scope("OutputLayer"):
        W3 = tf.Variable(tf.random_normal([n_hidden2, n_output]), name="W3")
        tf.summary.histogram("Weights 3", W3)

        b3 = tf.Variable(tf.random_uniform([n_output]), name="B3")
        tf.summary.histogram("Biases 3", b3)

        hy = tf.nn.softmax(tf.matmul(L2, W3) + b3)
        tf.summary.histogram("Output", hy)

    # calculate the coast of our calculations and then optimaze it
    with tf.name_scope("Coast"):
        # tf.reduce_mean(-Y * tf.log(hy) - (1 - Y) * tf.log(1 - hy))
        # tf.reduce_sum(tf.pow(Y_- hy, 2)) / (2 * train_size)
        # tf.sqrt(tf.reduce_mean(tf.squared_difference(Y, hy)))
        # tf.sqrt(tf.losses.mean_squared_error(hy, Y))
        cost = tf.sqrt(tf.losses.mean_squared_error(hy, Y))
        tf.summary.histogram("Cost ", cost)

    with tf.name_scope("Train"):
        optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)
        tf.summary.histogram("Optimazer ", optimizer.values())

    with tf.name_scope("accuracy"):
        answer = tf.equal(tf.floor(hy + 0.1), Y)
        accuracy = tf.reduce_mean(tf.cast(answer, "float32"))
        tf.summary.scalar("accuracy", accuracy)

    summ = tf.summary.merge_all()

    # Initialize variabls and tensorflow session
    sess.run(tf.global_variables_initializer())
    writer = tf.summary.FileWriter(LOGDIR)
    writer.add_graph(sess.graph)

    # lets Do  Our real traing
    saver = tf.train.Saver()
    with sess:
        for i in range(training_epochs):
            """
            xrange(training_epochs * train_size // BATCH_SIZE)
            offset = (i * BATCH_SIZE) % training_epochs
            batch_data = X_train[offset:(offset + BATCH_SIZE), :]
            batch_labels = y_train[offset:(offset + BATCH_SIZE)] """

            sess.run(optimizer, feed_dict={X: inputX, Y: inputY})
            print(sess.run(cost, feed_dict={X: inputX, Y: inputY}))
            if (i) % display_step == 0:
                cc = sess.run(cost, feed_dict={X: inputX, Y: inputY})
                print("Training step:", '%04d' % (i), "cost=", "{:.35f}".format(cc))
                save_path = saver.save(sess,"/home/benarousfarouk/Desktop/SSI/Anomaly-Detection-InLogFiles/Models/Demo2/model.ckpt")

    print("\n ------------------------------------Optimization "
          "Finished!------------------------------------------\n")
    print("Training cost=", cc,
          "\n W1 = \n", sess.run(W1), "\n W2= \n", sess.run(W2),
          "\n b1=", sess.run(b1), '\n', "\n b2=", sess.run(b2), '\n')

    answer = tf.equal(tf.floor(hy + 0.1), Y)
    accuracy = tf.reduce_mean(tf.cast(answer, "float32"))
    print("Accuracy: ", accuracy.eval({X: X_test, Y: y_test}, session=sess) * 100, "%")
    print("final Coast = ", cc)
    print("Parameters  :", "\n learning rate  = ", learning_rate, "\n epoches = ", training_epochs,
          " \n hidden layers  = ", n_hidden1, "\n coast function \n optimazer Adam ")

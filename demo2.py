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
from sklearn import metrics
import urllib.parse
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
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


def LoadNormalDataToCsv():
    file = open("Data/demo2/normalTrafficTraining.txt", "r")
    lines = file.readlines()

    TrafficList = []
    tmp = []
    for i in range(len(lines)):

        if i == len(lines) - 1:
            break

        if "GET http:" in lines[i] or "POST http:" in lines[i] or "http" in lines[i]:
            tmp.append(lines[i])
        else:
            # add a condition wich checks  if it belongs to each parameter
            if "modo" in lines[i]:
                continue

            V = lines[i].split(":", 1)
            if len(V) != 1:
                tmp.append(V[1])

        if "GET http" in lines[i + 1] or "POST http" in lines[i + 1] or "PUT http" in lines[i + 1] or "HEAD http" in \
                lines[i + 1] or "DELETE http" in lines[i + 1] or "CONNECT http" in lines[i + 1] or "OPTIONS http" in \
                lines[i + 1] or "TRACE http" in lines[i + 1] or "PATCH http" in lines[i + 1]:
            traffic = Traffic(tmp[0], tmp[1], tmp[2], tmp[3], tmp[4], tmp[5], tmp[6], tmp[7], tmp[8], tmp[9], tmp[10],
                              "0")
            TrafficList.append(traffic)
            del traffic
            tmp = []

        # Check if line is empty
        if not lines[i].strip():
            continue

    df = pd.DataFrame([t.__dict__ for t in TrafficList])

    df.to_csv(r'Data/demo2/normalTraffic.csv')
    return df


def LoadAbNormalDataToCsv():
    file = open("Data/demo2/anomalousTrafficTest.txt", "r")
    lines = file.readlines()

    TrafficList = []
    tmp = []
    for i in range(len(lines)):

        if i == len(lines) - 1:
            break

        if "GET http:" in lines[i] or "POST http:" in lines[i] or "http" in lines[i]:
            tmp.append(lines[i])
        else:
            # add a condition wich checks  if it belongs to each parameter
            if "modo" in lines[i]:
                continue

            V = lines[i].split(":", 1)
            if len(V) != 1:
                tmp.append(V[1])

        if "modo=" in lines[i]:
            continue
        if "GET http" in lines[i + 1] or "POST http" in lines[i + 1] or "PUT http" in lines[i + 1] or "HEAD http" in \
                lines[i + 1] or "DELETE http" in lines[i + 1] or "CONNECT http" in lines[i + 1] or "OPTIONS http" in \
                lines[i + 1] or "TRACE http" in lines[i + 1] or "PATCH http" in lines[i + 1]:
            traffic = Traffic(tmp[0], tmp[1], tmp[2], tmp[3], tmp[4], tmp[5], tmp[6], tmp[7], tmp[8], tmp[9], tmp[10],
                              "1")
            TrafficList.append(traffic)
            del traffic
            tmp = []

        # Check if line is empty
        if not lines[i].strip():
            continue

    df = pd.DataFrame([t.__dict__ for t in TrafficList])

    df.to_csv(r'Data/demo2/AbnormalTraffic.csv')
    return df


def LoadNormalTestToCsv():
    file = open("Data/demo2/normalTrafficTest.txt", "r")
    lines = file.readlines()

    TrafficList = []
    tmp = []
    for i in range(len(lines)):

        if i == len(lines) - 1:
            break

        if "GET http:" in lines[i] or "POST http:" in lines[i] or "http" in lines[i]:
            tmp.append(lines[i])
        else:
            # add a condition wich checks  if it belongs to each parameter
            if "modo" in lines[i]:
                continue

            V = lines[i].split(":", 1)
            if len(V) != 1:
                tmp.append(V[1])

        if "GET http" in lines[i + 1] or "POST http" in lines[i + 1] or "PUT http" in lines[i + 1] or "HEAD http" in \
                lines[i + 1] or "DELETE http" in lines[i + 1] or "CONNECT http" in lines[i + 1] or "OPTIONS http" in \
                lines[i + 1] or "TRACE http" in lines[i + 1] or "PATCH http" in lines[i + 1]:
            traffic = Traffic(tmp[0], tmp[1], tmp[2], tmp[3], tmp[4], tmp[5], tmp[6], tmp[7], tmp[8], tmp[9], tmp[10],
                              "0")
            TrafficList.append(traffic)
            del traffic
            tmp = []

        # Check if line is empty
        if not lines[i].strip():
            continue

    df = pd.DataFrame([t.__dict__ for t in TrafficList])

    df.to_csv(r'Data/demo2/normalTestTraffic.csv')
    return df


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


def ConvertCsvSmallAscii(dataframe):
    TrafficList = []
    tmp = []
    Target = []
    rounding = 1e+261
    # add the lenghth in the e
    for i, row in dataframe.iterrows():
        trafic = Traffic(row["http"], row["Agent"], row["Pragma"], row["Cache"], row["Accept"], row["Encoding"],
                         row["Charset"], row["Language"], row["Host"], row["Cookie"], row["Connection"], row["Target"])

        http = float(trafic.http)
        tmp = str(http).split("e", 1)
        http = float(tmp[0])
        http = round(http, 5)
        http = scaleBetween(http)
        trafic.http = str(http)

        agent = float(trafic.Agent)
        tmp = str(agent).split("e", 1)
        agent = float(tmp[0])
        agent = round(agent, 5)
        agent = scaleBetween(agent)
        trafic.Agent = str(agent)

        pragma = float(trafic.Pragma)
        tmp = str(pragma).split("e", 1)
        pragma = float(tmp[0])
        pragma = round(pragma, 5)
        pragma = scaleBetween(pragma)
        trafic.Pragma = str(pragma)

        cache = float(trafic.Cache)
        tmp = str(cache).split("e", 1)
        cache = float(tmp[0])
        cache = round(cache, 5)
        cache = scaleBetween(cache)
        trafic.Cache = str(cache)

        accept = float(trafic.Accept)
        tmp = str(accept).split("e", 1)
        accept = float(tmp[0])
        accept = round(accept, 5)
        accept = scaleBetween(accept)
        trafic.Accept = str(accept)

        encoding = float(trafic.Encoding)
        tmp = str(encoding).split("e", 1)
        encoding = float(tmp[0])
        encoding = round(encoding, 5)
        encoding = scaleBetween(encoding)
        trafic.Encoding = str(encoding)

        charset = float(trafic.Charset)
        tmp = str(charset).split("e", 1)
        charset = float(tmp[0])
        charset = round(charset, 5)
        charset = scaleBetween(charset)
        trafic.Charset = str(charset)

        language = float(trafic.Language)
        tmp = str(language).split("e", 1)
        language = float(tmp[0])
        language = round(language, 5)
        language = scaleBetween(language)
        trafic.Language = str(language)

        host = float(trafic.Host)
        tmp = str(host).split("e", 1)
        host = float(tmp[0])
        host = round(host, 5)
        host = scaleBetween(host)
        trafic.Host = str(host)

        coockie = float(trafic.Cookie)
        tmp = str(coockie).split("e", 1)
        coockie = float(tmp[0])
        coockie = round(coockie, 5)
        coockie = scaleBetween(coockie)
        trafic.Cookie = str(coockie)

        connection = float(trafic.Connection)
        tmp = str(connection).split("e", 1)
        connection = float(tmp[0])
        connection = round(connection, 5)
        connection = scaleBetween(connection)
        trafic.Connection = str(connection)

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
    dataframe.to_csv(r'Data/demo2/AllTrafficFinal.csv')

    return dataframe


if __name__ == "__main__":
    # for the winwods users you should change this path
    LOGDIR = "/tmp/mnist_tutorial/"

    print("Loading Normal Data ....")
    # dfNormal = LoadNormalDataToCsv()
    print("Loading AbNormal Data ....")
    # dfabnoraml = LoadAbNormalDataToCsv()
    print("Loading Test Data ....")
    # dfNormalTest = LoadNormalTestToCsv()

    print("Concatinating all the files into one   ....")
    # Dataframe = [dfNormal, dfabnoraml, dfNormalTest]
    # AllTrafics = pd.concat(Dataframe)
    # AllTrafics.to_csv(r'Data/demo2/AllTraffic.csv')

    print("Converting data  to ASCII represantation ...")
    # dataframe = pd.read_csv('Data/demo2/AllTraffic.csv')
    # df = ConvertCsvAscii(dataframe)
    # df.to_csv(r'Data/demo2/AllTrafficASCII.csv')

    # Convert to small numbers
    print("Converting data  to ASCII Small represantation ...")
    # dataframe = pd.read_csv("Data/demo2/AllTrafficASCII.csv")
    # df = ConvertCsvSmallAscii(dataframe)
    # df.to_csv(r'Data/demo2/AllTrafficSmallASCII.csv')

    # df = pd.read_csv("Data/demo2/AllTrafficSmallASCII.csv")
    # df =deleteinfvalue(df)

    df = pd.read_csv("Data/demo2/AllTrafficFinal.csv")
    print("Data is ready  , the training will start after a while")

    scaler = MinMaxScaler()
    df[['http', 'Agent', 'Pragma', 'Cache', 'Accept', 'Encoding', 'Charset', 'Language', 'Host', 'Cookie',
        'Connection']] = scaler.fit_transform(df[['http', 'Agent', 'Pragma', 'Cache', 'Accept', 'Encoding', 'Charset',
                                                  'Language', 'Host', 'Cookie', 'Connection']])

    df = pd.DataFrame.from_records(df)

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
    learning_rate = 0.1
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

        L1 = tf.nn.leaky_relu(tf.matmul(X, W1) + b1)
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
    with tf.name_scope("Cost"):
        # tf.reduce_mean(-Y * tf.log(hy) - (1 - Y) * tf.log(1 - hy))
        # tf.reduce_sum(tf.pow(Y_- hy, 2)) / (2 * train_size)
        # tf.sqrt(tf.reduce_mean(tf.squared_difference(Y, hy)))
        # tf.sqrt(tf.losses.mean_squared_error(hy, Y))
        cost = tf.reduce_sum(tf.pow(Y - hy, 2)) / (2 * train_size)
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
    """
                
                offset = (i * BATCH_SIZE) % training_epochs
                batch_data = X_train[offset:(offset + BATCH_SIZE), :]
                batch_labels = y_train[offset:(offset + BATCH_SIZE)] 
    """

    # lets Do  Our real traing
    saver = tf.train.Saver()
    with sess:
        # saver.restore(sess,"/home/benarousfarouk/Desktop/SSI/Anomaly-Detection-InLogFiles/Models/Demo2/model.ckpt")
        for i in xrange(training_epochs * train_size // BATCH_SIZE):

            offset = (i * BATCH_SIZE) % training_epochs
            batch_data = X_train[offset:(offset + BATCH_SIZE), :]
            batch_labels = y_train[offset:(offset + BATCH_SIZE)]
            sess.run(optimizer, feed_dict={X: X_train, Y: y_train})

            if (i) % display_step == 0:
                cc = sess.run(cost, feed_dict={X: X_train, Y: y_train})
                print("Training step:", '%04d' % (i), "cost=", "{:.35f}".format(cc))
                save_path = saver.save(sess,"/home/benarousfarouk/Desktop/SSI/Anomaly-Detection-InLogFiles/Models/Demo2/model.ckpt")

        print("Training cost=", cc, "\n W1 = \n", sess.run(W1), "\n W2= \n", "\n b1=", sess.run(b1), '\n', '\n')

        answer = tf.equal(tf.floor(hy + 0.1), Y)
        accuracy = tf.reduce_mean(tf.cast(answer, "float32"))
        print("Accuracy: ", accuracy.eval({X: X_train, Y: y_train}, session=sess) * 100, "%")
        print("final Coast = ", cc)
        print("Parameters  :", "\n learning rate  = ", learning_rate, "\n epoches = ", training_epochs,
              " \n hidden layers  = ", n_hidden1, "\n coast function \n optimazer Adam ")

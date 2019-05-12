import dataframe as dataframe
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pandas import Series
from sklearn.feature_extraction.text import TfidfVectorizer
import os
from sklearn.cross_decomposition import tests
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import urllib.parse
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


class Traffic:
    def __init__(self, http, Agent, Pragma, Cache, Accept, Encoding, Charset, Language, Host, Cookie, Connection):
        self.http = http
        self.Agent = Agent
        self.Pragma = Pragma
        self.Cache = Cache
        self.Accept = Accept
        self.Encoding = Encoding
        self.Charrset = Charset
        self.Language = Language
        self.Host = Host
        self.Cookie = Cookie
        self.Connection = Connection


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
            traffic = Traffic(tmp[0], tmp[1], tmp[2], tmp[3], tmp[4], tmp[5], tmp[6], tmp[7], tmp[8], tmp[9], tmp[10])
            TrafficList.append(traffic)
            del traffic
            tmp = []

        # Check if line is empty
        if not lines[i].strip():
            continue

    df = pd.DataFrame([t.__dict__ for t in TrafficList])
    Target = []
    for i in range(len(df)):
        Target.append("0")
    # Target = np.reshape(Target, (1,len(df)-1)).T

    df["Target"] = Series(Target, index=df.index)

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
            traffic = Traffic(tmp[0], tmp[1], tmp[2], tmp[3], tmp[4], tmp[5], tmp[6], tmp[7], tmp[8], tmp[9], tmp[10])
            TrafficList.append(traffic)
            del traffic
            tmp = []

        # Check if line is empty
        if not lines[i].strip():
            continue

    df = pd.DataFrame([t.__dict__ for t in TrafficList])
    Target = []
    for i in range(len(df)):
        Target.append("1")

    df["Target"] = Series(Target, index=df.index)

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
            traffic = Traffic(tmp[0], tmp[1], tmp[2], tmp[3], tmp[4], tmp[5], tmp[6], tmp[7], tmp[8], tmp[9], tmp[10])
            TrafficList.append(traffic)
            del traffic
            tmp = []

        # Check if line is empty
        if not lines[i].strip():
            continue

    df = pd.DataFrame([t.__dict__ for t in TrafficList])
    Target = []
    for i in range(len(df)):
        Target.append("0")
    df["Target"] = Series(Target, index=df.index)

    df.to_csv(r'Data/demo2/normalTestTraffic.csv')
    return df


if __name__ == "__main__":
    # for the winwods users you should change this path
    LOGDIR = "/tmp/mnist_tutorial/"

    print("Loading Normal Data ....")
    dfNormal = LoadNormalDataToCsv()
    print(dataframe)
    print("Loading AbNormal Data ....")
    dfabnoraml = LoadAbNormalDataToCsv()
    print("Loading Test Data ....")
    dfNormalTest = LoadNormalTestToCsv()
    print("Data is ready ")

    Dataframe = [dfNormal, dfabnoraml, dfNormalTest]
    AllTrafics = pd.concat(Dataframe)

    AllTrafics.to_csv(r'Data/demo2/AllTraffic.csv')

    print(AllTrafics)

    dataframe = pd.read_csv('Data/demo2/AllTraffic.csv')

    inputX = dataframe.loc[:,
             ['http', 'Agent', 'Pragma', 'Cache', 'Accept', 'Encoding', ' Charset', 'Language', 'Host', 'Cookie',
              'Connection']].values

    inputY = dataframe.loc[:, ["Target"]].values

    X_train, X_test, y_train, y_test = train_test_split(inputX, inputY, test_size=0.2, random_state=42)  # splitting data

    # Parameters
    n_input = 11  # features
    n_hidden = 7  # hidden nodes
    n_output = 1  # lables
    learning_rate = 0.001
    training_epochs = 1000000  # simply iterations
    display_step = 10000  # to split the display
    # n_samples = inputY.size  # number of the instances

    sess = tf.Session()

    X = tf.placeholder(tf.float32, name="X")
    tf.summary.histogram("inputs ", X)

    Y = tf.placeholder(tf.float32, name="output")
    tf.summary.histogram("outputs ", Y)

    with tf.name_scope("Hidden_Layer"):
        W1 = tf.Variable(tf.zeros([n_input, n_hidden]), name="W1")
        tf.summary.histogram("Weights 1", W1)
        b1 = tf.Variable(tf.zeros([n_hidden]), name="B1")
        tf.summary.histogram("Biases 1", b1)
        L2 = tf.nn.sigmoid(tf.matmul(X, W1) + b1)
        tf.summary.histogram("Activation", L2)

    with tf.name_scope("OutputLayer"):
        W2 = tf.Variable(tf.zeros([n_hidden, n_output]), name="W2")
        tf.summary.histogram("Weights 2", W2)
        b2 = tf.Variable(tf.zeros([n_output]), name="B2")
        tf.summary.histogram("Biases 2", b2)
        hy = tf.nn.sigmoid(tf.matmul(L2, W2) + b2)
        tf.summary.histogram("Output", hy)

    # calculate the coast of our calculations and then optimaze it
    with tf.name_scope("Coast"):
        cost = tf.reduce_mean(-Y * tf.log(hy) - (1 - Y) * tf.log(1 - hy))
        tf.summary.histogram("Cost ", cost)

    with tf.name_scope("Train"):
        optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)
        tf.summary.histogram("Optimazer ", optimizer.values())

    with tf.name_scope("accuracy"):
        answer = tf.equal(tf.floor(hy + 0.1), Y)
        accuracy = tf.reduce_mean(tf.cast(answer, "float32"))
        tf.summary.scalar("accuracy", accuracy)

    summ = tf.summary.merge_all()
    saver = tf.train.Saver()
    """cost = tf.reduce_sum(tf.pow(y_ - y, 2)) / (2 * n_samples)
      optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)
     """
    # Initialize variabls and tensorflow session
    sess.run(tf.global_variables_initializer())
    writer = tf.summary.FileWriter(LOGDIR)
    writer.add_graph(sess.graph)

    # lets Do  Our real traing

    for i in range(training_epochs):
        sess.run(optimizer, feed_dict={X: X_train, Y: y_train})
        # Take a gradient descent step using our inputs and  labels

        # That's all! The rest of the cell just outputs debug messages.
        # Display logs per epoch step

        if (i) % display_step == 0:
            cc = sess.run(cost, feed_dict={X: X_train, Y: y_train})
            print("Training step:", '%04d' % (i), "cost=", "{:.35f}".format(cc))
            # print("\n  W1=", sess.run(W1), " \n W1=", sess.run(W2),
            # "\n b1=", sess.run(b1), "b2=", sess.run(b2) )

    print("\n ------------------------------------Optimization "
          "Finished!------------------------------------------\n")
    training_cost = cc
    print("Training cost=", training_cost,
          "\n W1 = \n", sess.run(W1), "\n W2= \n", sess.run(W2),
          "\n b1=", sess.run(b1), '\n', "\n b2=", sess.run(b2), '\n')

    answer = tf.equal(tf.floor(hy + 0.1), Y)
    accuracy = tf.reduce_mean(tf.cast(answer, "float32"))
    # print(sess.run([hy], feed_dict={X: inputX, Y: inputY}))
    print("Accuracy: ", accuracy.eval({X: X_test, Y: y_test}, session=sess) * 100, "%")
    print("final Coast = ", training_cost)
    print("Parameters  :", "\n learning rate  = ", learning_rate, "\n epoches = ", training_epochs,
          " \n hidden layers  = ", n_hidden, "\n coast function \n optimazer Adam ")

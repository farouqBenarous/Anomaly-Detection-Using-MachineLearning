import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


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


if __name__ == "__main__":
    TrafficList = []
    # Let's have Pandas load our dataset as a dataframe
    """ 
    dataframe = pd.read_csv("Data/demo2/normalTrafficTraining.txt" , delimiter=" \n" , header=None )
    DfSize = dataframe.shape[0]
    """

    file = open("Data/demo2/normalTrafficTraining.txt", "r")
    lines = file.readlines()

    TrafficList = []
    tmp = []
    for i in range(len(lines)):
        tmp.append(lines[i])

        if i == len(lines) - 1:
            break

        if "GET http:" in lines[i + 1] or "POST http:" in lines[i + 1] or "http" in lines[i + 1]:
            traffic = Traffic(tmp[0], tmp[1], tmp[2], tmp[3], tmp[4], tmp[5], tmp[6], tmp[7], tmp[8], tmp[9], tmp[10])
            TrafficList.append(traffic)
            print(traffic.http)
            print(" \n")
            print(len(TrafficList))
            del traffic
            tmp = []

    print(TrafficList[0].http)
    print("\n")
    print(TrafficList[1].http)
    # Let's prepare some parameters for the training process

    # Parameters
    n_input = 13  # features
    n_hidden = 7  # hidden nodes
    n_output = 1  # lables
    learning_rate = 0.001
    training_epochs = 1000000  # simply iterations
    display_step = 10000  # to split the display
    # n_samples = inputY.size  # number of the instances


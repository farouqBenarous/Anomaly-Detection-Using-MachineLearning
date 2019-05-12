import dataframe as dataframe
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pandas import Series


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


    # Parameters
    n_input = 13  # features
    n_hidden = 7  # hidden nodes
    n_output = 1  # lables
    learning_rate = 0.001
    training_epochs = 1000000  # simply iterations
    display_step = 10000  # to split the display
    # n_samples = inputY.size  # number of the instances

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


if __name__ == "__main__":
    # Let's have Pandas load our dataset as a dataframe
    dataframe = pd.read_csv("datasetcsv.csv")
    # remove columns we don't care about

    inputX = dataframe.loc[:,
             ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca',
              'thal']].values

    inputY = dataframe.loc[:, ["target"]].values

    # Let's prepare some parameters for the training process

    # Parameters
    n_input = 13  # features
    n_hidden = 7  # hidden nodes
    n_output = 1  # lables
    learning_rate = 0.001
    training_epochs = 1000000  # simply iterations
    display_step = 10000  # to split the display
    n_samples = inputY.size  # number of the instances

    tf.reset_default_graph()
    sess = tf.Session()
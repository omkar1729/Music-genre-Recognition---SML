import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn import metrics

# Preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler


import warnings
warnings.filterwarnings('ignore')


# Dataset Path
DATASET_PATH = "data.csv"


def main():
    # Music dataset headers
    music_data_headers = ["id", "chroma_stft", "rmse", "spectral_centroid", "spectral_bandwidth",
                          "rolloff", "zero_crossing_rate", "label"]
    # Loading the Music dataset in to Pandas dataframe
    music_data = pd.read_csv(DATASET_PATH, usecols=music_data_headers)

    # Dropping unneccesary columns
    #music_data = music_data.drop(['filename'], axis=1)

    print("Number of observations :: " + str(len(music_data.index.values)))
    print("Number of columns :: " + str(len(music_data.columns)))
    print("Headers :: " + str(music_data.columns.values))

    train_x, test_x, train_y, test_y = train_test_split(music_data[music_data_headers[:-1]],
                                                        music_data[music_data_headers[-1]], train_size=0.7)

    # Train multi-classification model with logistic regression
    lr = linear_model.LogisticRegression()
    lr.fit(train_x, train_y)

    # Train multinomial logistic regression
    mul_lr = linear_model.LogisticRegression(multi_class='multinomial', solver='newton-cg').fit(train_x, train_y)

    print("Logistic regression Train Accuracy :: ", metrics.accuracy_score(train_y, lr.predict(train_x)))
    print("Logistic regression Test Accuracy :: ", metrics.accuracy_score(test_y, lr.predict(test_x)))

    print("Multinomial Logistic regression Train Accuracy :: ",
          metrics.accuracy_score(train_y, mul_lr.predict(train_x)))
    print("Multinomial Logistic regression Test Accuracy :: ", metrics.accuracy_score(test_y, mul_lr.predict(test_x)))

if __name__ == "__main__":
    main()
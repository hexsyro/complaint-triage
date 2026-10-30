
import pandas as pd
import time
import joblib

from scipy.sparse import load_npz
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV



def training(x_train, y_train):
    s_time=time.time()
    m=CalibratedClassifierCV(LinearSVC(), cv=3).fit(x_train,y_train)
    e_time=time.time()

    train_time =(e_time-s_time)%60
    return train_time, m


def main():
    x_train=load_npz('data/processed/x_train.npz')
    y_train=pd.read_csv('data/processed/y_train.csv').squeeze('columns')
    print()
    print()
    print("input and target files loading completed successfully!")

    train_time, m=training(x_train, y_train)
    print(f"model training completed successfully within {train_time:.2f} seconds.")
    print()
    joblib.dump(m, 'models/LinearSVC_model.joblib')
    print("model file exporting completed successfully!")


if __name__ == '__main__':
    main()
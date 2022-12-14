import pandas as pd 
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os


def prepare_data(filename):
    df_proc = pd.read_csv(os.path.join('data/processed', filename))
    X = df_proc.drop('Outcome', axis = 1)
    X = X.values
    y = df_proc['Outcome']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)
    return X_train, X_test, y_train, y_test


def load_model():
    path = 'models/xgbc_model.json'
    model = joblib.load(path)
    print("Model loaded correctly")
    return model


def conf_matrix(y_test, y_pred, label_dict):
    results = pd.DataFrame({'Predict': y_pred,'Target': y_test})
    cm = confusion_matrix(y_pred=results['Predict'], y_true=results['Target'])
    cmn = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    sns.heatmap(cmn,annot=True,xticklabels=label_dict.values()
                ,yticklabels=label_dict.values(),cmap='plasma')
    plt.xlabel('Predicted')
    plt.ylabel('Target')
    plt.savefig('data/scores/model_conf_matrix.png', dpi=120)


def main():
    X_train, X_test, y_train, y_test = prepare_data('diabetes.csv')
    model = load_model()
    y_pred = model.predict(X_test)
    label_dict = {0: 'healthy', 1: 'diabetic'}
    print("Confusion matrix saved in data scores folder")
    conf_matrix(y_test, y_pred, label_dict)


if __name__ == "__main__":
    main()
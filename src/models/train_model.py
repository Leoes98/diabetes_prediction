import pandas as pd 
from sklearn.model_selection import train_test_split
import xgboost as xgb
import joblib
import os


def prepare_data(filename):
    df_proc = pd.read_csv(os.path.join('data/processed', filename))
    X = df_proc.drop('Outcome', axis = 1)
    X = X.values
    y = df_proc['Outcome']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)
    return X_train, X_test, y_train, y_test


def train_model(filename):
    X_train, X_test, y_train, y_test = prepare_data(filename)
    xgbc=xgb.XGBClassifier(max_depth=2, n_estimators=50, objective='binary:logistic', seed=0, subsample=.8, scale_pos_weight=2)
    xgbc.fit(X_train, y_train)    
    return xgbc


def save_model(model):
    path = 'models/xgbc_model.json'
    joblib.dump(model, path)
    print("Model saved correctly in models folder")


def main():
    model = train_model('diabetes.csv')
    save_model(model)
    print("Training process finished")


if __name__ == "__main__":
    main()
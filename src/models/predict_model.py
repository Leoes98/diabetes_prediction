import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
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


def class_report(y_test, y_pred):
    print(classification_report(y_test, y_pred))

    with open('data/scores/metrics.txt', 'w') as outfile:
        outfile.write(f'\nClassification Report:\n{classification_report(y_test, y_pred)}')

def main():
    X_train, X_test, y_train, y_test = prepare_data('diabetes.csv')
    model = load_model()
    y_pred = model.predict(X_test)
    print("Classification Report")
    class_report(y_test, y_pred)


if __name__ == "__main__":
    main()
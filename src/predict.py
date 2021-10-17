import joblib
import pandas as pd
from sklearn.metrics import classification_report

if __name__ == "__main__":
    model = joblib.load("Model/xgb.model")
    x_test = pd.read_csv("DataTest/x_test.csv")
    y_test = pd.read_csv("DataTest/y_test.csv")

    x_test.drop(columns=[x_test.columns[0]], inplace=True)
    y_test.drop(columns=[y_test.columns[0]], inplace=True)

    y_pred = model.predict(x_test)
    print(classification_report(y_test, y_pred))

    file = open("metrics.txt", 'w')
    file.write(classification_report(y_test, y_pred))

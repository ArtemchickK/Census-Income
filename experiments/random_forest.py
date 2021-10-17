import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
import os

if __name__ == "__main__":
    current_directory = os.getcwd()
    current_path = current_directory.split(sep="\\")
    project_path = "/".join(current_path[:-1])
    data_path = project_path + "/src/DataPreprocessed/data_preprocessed.csv"

    data = pd.read_csv(data_path)
    data.drop(columns=[data.columns[0]], inplace=True)

    y = data['income']
    X = data.drop(columns=['income'])

    categorical = X.select_dtypes(include='object').columns
    numeric = X.select_dtypes(include='int64').columns

    x_train, x_test, y_train, y_test = train_test_split(X, y, train_size=0.75, random_state=777)
    ct = ColumnTransformer([('ohe', OneHotEncoder(), categorical)], remainder='passthrough')
    pipe = Pipeline([('ct', ct), ('model', RandomForestClassifier())])

    pipe.fit(x_train, y_train)
    y_pred = pipe.predict(x_test)
    print(classification_report(y_test, y_pred))
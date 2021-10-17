import pandas as pd
import xgboost as xgb
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
import joblib


if __name__ == "__main__":
    data = pd.read_csv("DataPreprocessed/data_preprocessed.csv")
    data.drop(columns=[data.columns[0]], inplace=True)

    y = data['income']
    X = data.drop(columns=['income'])

    categorical = X.select_dtypes(include='object').columns
    numeric = X.select_dtypes(include='int64').columns
    X = pd.get_dummies(X, columns=categorical)
    x_train, x_test, y_train, y_test = train_test_split(X, y, train_size=0.75, random_state=777)

    # best parameters
    params = {'n_estimators': 170,
              'learning_rate': 0.15,
              'max_depth': 5,
              'min_child_weight': 1,
              'subsample': 0.8,
              'colsample_bytree': 0.8,
              'n_jobs': 4}

    model = xgb.XGBClassifier(**params)
    model.fit(x_train, y_train, eval_metric='auc', eval_set=[[x_train, y_train], [x_test, y_test]])

    x_test.to_csv("DataTest/x_test.csv")
    y_test.to_csv("DataTest/y_test.csv")
    joblib.dump(model, "Model/xgb.model")

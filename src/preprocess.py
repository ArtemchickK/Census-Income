import pandas as pd
import numpy as np

if __name__ == "__main__":
    data = pd.read_csv('Data/adult.csv')

    data['workclass'] = data['workclass'].replace(" ?", ' Private')
    data['occupation'] = data['occupation'].replace(" ?", ' Prof-specialty')
    data['native-country'] = data['native-country'].replace(" ?", ' United-States')

    data['marital-status'] = data['marital-status'].replace([' Married-civ-spouse', ' Married-AF-spouse'], 'Married')
    data['marital-status'] = data['marital-status'].replace(
        [' Divorced', ' Separated', ' Widowed', ' Married-spouse-absent'], 'Single')

    data['education'] = data['education'].replace(
        [' Preschool', ' 1st-4th', ' 5th-6th', ' 7th-8th', ' 9th', ' 10th', ' 11th', ' 12th'], 'school')
    data['education'] = data['education'].replace([' Assoc-voc', ' Assoc-acdm', ' Prof-school', ' Some-college'],
                                                  'Higher_eduction')

    data['income'] = data['income'].map({' <=50K': 0, ' >50K': 1})

    data.to_csv("DataPreprocessed/data_preprocessed.csv")
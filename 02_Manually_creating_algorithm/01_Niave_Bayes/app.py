import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

import sys
import os

class Naive_Bayes:
    def __init__(self):
        self.probabilities = None
        self.y_counts = None
        self.x_cols = None
    def fit(self,x,y):
        counts = y.squeeze().value_counts().to_dict()
        self.y_counts = counts
        self.x_cols = x.columns
        def calculating_probabilities(x,y):
            probabilities = {}
            
            x[y.columns] = y
            df = x
            for col in x.columns:
                result = (
                    df.groupby(col)
                    .agg(
                        true=(y.columns[0], lambda x: (x == 1).sum()),
                        false=(y.columns[0], lambda x: (x == 0).sum())
                    )
                    .reset_index()
                )

                result['Totaltrue'] = counts[1]
                result['Totalfalse'] = counts[0]

                num_values = len(result)

                result['p_true'] = (result['true'] + 1) / (result['Totaltrue'] + num_values)
                result['p_false'] = (result['false'] + 1) / (result['Totalfalse'] + num_values)
                
                probabilities[col] = result
            return probabilities
        self.probabilities = calculating_probabilities(x,y)
    def predict(self,x_predict):
        predict_values = []
        for i in range(0,len(x_predict)):
            total = self.y_counts[0] + self.y_counts[1]
            p_total_true=self.y_counts[1] / total
            p_total_false=self.y_counts[0] / total
            for j in range(0,len(self.x_cols)):
                key= self.x_cols[j]
                value= x_predict.iloc[i,j]
                row = self.probabilities[key][self.probabilities[key][key] == value]
                p_total_true *= row['p_true'].iloc[0]
                p_total_false *= row['p_false'].iloc[0]
            print('true',p_total_true)
            print('false',p_total_false)
            # return
            if p_total_true > p_total_false:
                predict_values.append(1)
            else:
                predict_values.append(0)

        return predict_values

def read_df(file_path):
    df = pd.read_csv(
    file_path,
    engine='pyarrow'
    )
    return df


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
file_path_train = os.path.join(BASE_DIR, "data", "cleaned", "data.csv")
file_path_predict = os.path.join(BASE_DIR, "data", "cleaned", "dataPredict.csv")

df = read_df(file_path_train)
df_predict = read_df(file_path_predict)

x_train = df.iloc[:,1:-1]
y_train = df.iloc[:,[-1]]

x_test = df_predict.iloc[:,1:-1]
y_test = df_predict.iloc[:,[-1]]

nb_model = Naive_Bayes()
nb_model.fit(x_train,y_train)
y_predict = nb_model.predict(x_test)
print(classification_report(y_test,y_predict))
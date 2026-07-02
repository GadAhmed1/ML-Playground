import pandas as pd

class Naive_Bayes:
    def __init__(self):
        pass
    def fit(self,x,y):
        counts = y.value_counts().to_dict()
        print(counts)

def read_df(file_path):
    df = pd.read_csv(
    file_path,
    engine='pyarrow'
    )
    return df

def calculating_probabilities(df,target_col_name,col):
    if target_col_name == col:
        result = (
            df.groupby('play')
            .agg(
                count=('play','count')
            )
            .reset_index()
        )
        return result
    else:
        result = df.groupby(col).agg(
        Yes=('play', lambda x: (x == 'Yes').sum()),
        No=('play', lambda x: (x == 'No').sum())
        ).reset_index()
        Total_Postives = get_y_counts(df,target_col_name)[0][1]
        Total_Nagtives = get_y_counts(df,target_col_name)[1][1]
        result['TotalYes']=Total_Postives
        result['TotalNo']=Total_Nagtives
        result['p_Yes']=result['Yes'] / result['TotalYes']
        result['p_No']=result['No'] / result['TotalNo']
        return result




df = read_df(r'D:\Main\projects\ML-playground-\02_Manually_creating_algorithm\01_Niave_Bayes\data.csv')
x_train = df.iloc[:,1:-2]
y_train = df.iloc[:,-1]
nb_model = Naive_Bayes()
nb_model.fit(x_train,y_train)
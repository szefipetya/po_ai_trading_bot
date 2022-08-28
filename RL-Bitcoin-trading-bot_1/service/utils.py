import os
from datetime import datetime
import numpy as np

def Write_to_file(Date, net_worth, filename='{}.txt'.format(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))):
    for i in net_worth: 
        Date += " {}".format(i)
    #print(Date)
    if not os.path.exists('logs'):
        os.makedirs('logs')
    file = open("logs/"+filename, 'a+')
    file.write(Date+"\n")
    file.close()

def Normalizing(df_original):
    df = df_original.copy()
    column_names = df.columns.tolist()
    for column in column_names[1:]:
        # Logging and Differencing
        test = np.log(df[column]) - np.log(df[column].shift(1))
        if test[1:].isnull().any():
            df[column] = df[column] - df[column].shift(1)
        else:
            df[column] = np.log(df[column]) - np.log(df[column].shift(1))
        # Min Max Scaler implemented
        Min = df[column].min()
        Max = df[column].max()
        df[column] = (df[column] - Min) / (Max - Min)

    return df
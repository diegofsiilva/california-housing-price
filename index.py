import pandas as pd
import matplotlib as plt
import seaborn as sns
import numpy as np
import sklearn as sk

data = pd.read_csv('housing.csv')
print(data.dropna(inplace=True)) ## quando eu coloco o dropna ele vai retirar todas as linhas 
## que contenham pelo menos um valor "non-value"

## o inplace=true ele mudar o dataframe original tirando todos valores non mas não vai fazer uma copia do anterior
data.info()
print("-------------------------------------")
from sklearn.model_selection import train_test_split
x = data.drop(['median_house_value'], axis=1)
y = data['median_house_value']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
train_data = x_train.join(y_train)
print(train_data)
print(train_data.hist())
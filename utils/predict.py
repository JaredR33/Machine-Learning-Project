#libraries
import pandas as pd
import datetime
import pickle
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import os 
#data


import os
os.chdir(os.path.dirname(__file__))
# script_dir = os.path.dirname(__file__) #<-- absolute dir the script is in
# rel_path = "data\data_big.csv"
# abs_file_path = os.path.join(script_dir, rel_path)
# df = pd.read_csv(abs_file_path, sep=';')
df = pd.read_csv("data/data_big.csv", sep=';')
X = df.drop(columns=['Target'])
y = df['Target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state=33)

#model




pipe_final = Pipeline(steps=[
                    ('scaler', StandardScaler()),
                    ('pca', PCA(n_components=9)),
                    ('classifier', RandomForestClassifier(max_depth=6, min_samples_leaf=2))
])
pipe_final.fit(X_train,y_train)

#saving

data_path = 'data'

# Guardar el archivo en la carpeta data
file_path = os.path.join(data_path, 'predicciones_test.csv')

y_pred_best = pipe_final.predict(X_test)
y_pred_best = pd.Series(y_pred_best)
y_pred_best.to_csv(file_path)
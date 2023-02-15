#libraries
import pandas as pd
import datetime
import pickle
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

#data

df = pd.read_csv('../data/data_big.csv', sep=';')
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


fecha = datetime.datetime.now().strftime('%y%m%d%H%M%S')
nombre_modelo = f"model_{fecha}"

with open(nombre_modelo, 'wb') as archivo_salida:
    pickle.dump(pipe_final, archivo_salida)
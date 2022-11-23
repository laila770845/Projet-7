# -*- coding: utf-8 -*-

# Import des librairies
from flask import Flask, jsonify
import sklearn
from sklearn import preprocessing

import pandas as pd
import pickle
from lightgbm import LGBMClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

app = Flask(__name__)

#Chargement de la base de données
#data = pd.read_csv('Clients_test.csv')
data = pd.read_csv('Clients_test_dashboard.csv')
data_train = pd.read_csv('Clients_train.csv')

#Chargement du modèle
model = pickle.load(open('model_best_lgb.pkl', 'rb'))

#Chargement du preprocessor
preprocessor = pickle.load(open('preprocessor.pkl', 'rb'))

@app.route('/')
def hello():
    return 'Hello, World!'

#Création pipeline
# On retire les variables TARGET pour ne pas les standardiser
data_stand = data.drop(['SK_ID_CURR'], axis=1)

# Isoler les variables binaires pour standardisation
col_binary = data_stand.columns[data_stand.isin([0,1]).all()]
data_non_binary = data_stand.drop(col_binary, axis=1)
col=data_non_binary.columns
data_binary = data_stand[col_binary]

#On définit une Pipeline à nos variables numériques, i.e. une chaine de transformation que nos variables subissent
#Pipeline des variables binaires
pipeline_binaire = Pipeline([('imputer', SimpleImputer(strategy="median", fill_value="missing"))])

#Pipeline des variables non binaires
pipeline_non_binaire = Pipeline([('imputer', SimpleImputer(strategy='median', fill_value='missing')),
                        ('scaler', StandardScaler())])

#Column_transformer permet d'appliquer nos transformers sur les colonnes que nous sélectionnons
#On obtient un transformer que l'on nomme preprocessor
preprocess=ColumnTransformer(transformers=[('binaire', pipeline_binaire, col_binary),
                                             ('non_binaire', pipeline_non_binaire, col)])

X_train_fit = preprocessor.fit(data_train)

@app.route('/prediction/<identifiant>')
def prediction(identifiant):
    print('identifiant du client = ', identifiant)

    # Récupération des données du client en question
    ID = int(identifiant)
    X = data[data['SK_ID_CURR'] == ID]
    X_sans_id = X.drop(columns='SK_ID_CURR')
    X_pred = X_train_fit.transform(X_sans_id)
    proba = model.predict_proba(X_pred)
    pred = model.predict(X_pred)

    # DEBUG
    # print('id_client : ', id_client)

    dict_final = {
        'prediction': int(pred),
        'proba': float(proba[0][0])
    }

    print('Nouvelle Prédiction : \n', dict_final)

    return jsonify(dict_final)


# lancement de l'application
if __name__ == "__main__":
    app.run(debug=True)
import streamlit as st
import streamlit.components.v1 as components
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pyplot import *
import seaborn as sns
import pickle
import time
import math
from urllib.request import urlopen
import json
import requests
import plotly
import plotly.graph_objects as go
import shap
import plotly.express as px
from PIL import Image

import warnings

warnings.filterwarnings("ignore")

@st.cache(allow_output_mutation=True)
def load_data():
    # Chargement de la base de données test
    #data = pd.read_csv('Clients_test.csv')
    data = pd.read_csv('Clients_test_dashboard.csv')


    # description des features
    description = pd.read_csv('HomeCredit_columns_description.csv',
                              usecols=['Row', 'Description'],
                              index_col=0, encoding='unicode_escape')

    return data, description

@st.cache(allow_output_mutation=True)
def load_model():
    #Chargement du modèle
    return pickle.load(open('model_best_lgb.pkl', 'rb'))

@st.cache(allow_output_mutation=True)
def load_shap_values():
    # Chargement du shap_values
    return pickle.load(open('shap_values', 'rb'))

@st.cache(allow_output_mutation=True)
def load_explainer_value():
    # Chargement des valeurs du explainer
    return pickle.load(open('explainer_expected_value', 'rb'))

@st.cache(allow_output_mutation=True)
def load_explainer():
    # Chargement du explainer
    return pickle.load(open('explainer', 'rb'))

@st.cache
def load_infos_gen(data):
    lst_infos = [data.shape[0],
                 round(data["AMT_INCOME_TOTAL"].mean(), 2),
                 round(data["AMT_CREDIT"].mean(), 2)]

    nb_credits = lst_infos[0]
    rev_moy = lst_infos[1]
    credits_moy = lst_infos[2]
    targets = data.TARGET.value_counts()

    return nb_credits, rev_moy, credits_moy, targets

#
@st.cache
def load_age_population(data):
    data_age = round((data["DAYS_BIRTH"] / -365), 2)
    return data_age

#
@st.cache
def load_income_population(sample):
    data_income = pd.DataFrame(sample["AMT_INCOME_TOTAL"])
    data_income = data_income.loc[data_income['AMT_INCOME_TOTAL'] < 200000, :]
    return data_income

@st.cache
def get_client_info(data, id_client):
    client_info = data[data['SK_ID_CURR'] == int(id_client)]
    #client_info = data.iloc[id_client]
    return client_info

#@st.cache
def graphique(data,variable, client_val, titre):

    if (not (math.isnan(client_val))):
        fig = plt.figure(figsize = (10, 4))

        negatif = data.loc[data['TARGET'] == 0]
        positif = data.loc[data['TARGET'] == 1]

        if (variable == "DAYS_BIRTH"):
            sns.kdeplot((negatif[variable]/-365).dropna(), label = 'Rembourse', color='g')
            sns.kdeplot((positif[variable]/-365).dropna(), label = 'Ne rembourse pas', color='r')
            plt.axvline(float(client_val/-365), \
                        color="blue", linestyle='--', label = 'Valeur Client')


        else:
            sns.kdeplot(negatif[variable].dropna(), label = 'Rembourse', color='g')
            sns.kdeplot(positif[variable].dropna(), label = 'Ne rembourse pas', color='r')
            plt.axvline(float(client_val), color="blue", \
                        linestyle='--', label = 'Valeur Client')


        plt.title(titre, fontsize='20', fontweight='bold')
        plt.legend()
        plt.show()
        st.pyplot(fig)
    else:
        st.write("Il n'y a pas de valeurs pour cette variable ")


# Chargement des données
data, description = load_data()

# Chargement de la base de données application train pour les infos client
data_info = pd.read_csv('Clients_data_dashboard.csv')

# Chargement du modèle
model = load_model()

# Chargement du preprocessor
preprocessor = pickle.load(open('preprocessor.pkl', 'rb'))

#Chargement des valeurs attendues du explainer
explainer_exp_value = load_explainer_value()

# Chargement de explainer
explainer = load_explainer()


# @st.cache
def st_shap(plot, height=None):
    shap_html = f"<head>{shap.getjs()}</head><body>{plot.html()}</body>"
    components.html(shap_html, height=height)

#
# SIDEBAR
#

LOGO_IMAGE = "logo.png"
SHAP_GENERAL = "global_feature_importance.png"

with st.sidebar:
    st.image(LOGO_IMAGE,width=200)

    show_credit_decision = st.checkbox("Décision de crédit")
    show_client_details = st.checkbox("Informations descriprives relatives à ce client")
    show_client_comparison = st.checkbox("Comparaison aux autres clients")
    shap_general = st.checkbox("Feature importance globale")
    if (st.checkbox("Description des variables")):
        list_var = description.index.to_list()
        list_var = list(dict.fromkeys(list_var))
        var = st.selectbox('Sélectionner une variable', \
                               sorted(list_var))

        desc = description['Description'].loc[description.index == var][:1]
        st.markdown('**{}**'.format(desc.iloc[0]))

#
# PARTIE En tête
#

# Titre :
st.markdown("<h1 style='text-align: center; color: #5A5E6B;'>DOSSIER CLIENT : </h1>", unsafe_allow_html=True)

# Afficher l'ID Client sélectionné
st.write(" ")
id_list = data["SK_ID_CURR"].tolist()
id_list = sorted(id_list)
id_client = st.selectbox(" Sélectionnez l'identifiant du client ", id_list)

# Ligne de séparation :
st.markdown("***")

if (int(id_client) in id_list):

    client_info = get_client_info(data_info, id_client)


    #
    # Partie décision de crédit
    #

    if (show_credit_decision):
        st.markdown("<h2 style='text-align: center; color: #5A5E6B;'>Décision de crédit</h2>",
            unsafe_allow_html=True)

        "Utilisation de l'API :"

        API_lien = "https://api-projet7.herokuapp.com/prediction/" + str(id_client)

        with st.spinner('Chargement du score du client...'):
            json_lien = urlopen(API_lien)

            API_data = json.loads(json_lien.read())
            prediction = API_data['prediction']
            proba = API_data['proba']
            proba = round(proba * 100, 1)

            #gauge
            gauge = go.Figure(go.Indicator(
                mode="gauge+delta+number",
                title={'text': 'Probabilité de remboursement'},
                value= proba,
                domain={'x': [0, 1], 'y': [0, 1]},
                gauge={'axis': {'range': [None, 100]},
                       'steps': [
                           {'range': [0, 60], 'color':"darksalmon" },
                           {'range': [60, 100], 'color': "lightgreen"},
                       ],
                       'threshold': {
                           'line': {'color': "white", 'width': 10},
                           'thickness': 0.8,
                           'value': proba},

                       'bar': {'color': "white", 'thickness': 0.2},
                       },
            ))

            gauge.update_layout(width=450, height=250,
                                margin=dict(l=50, r=50, b=0, t=0, pad=4))
            #Résultat demande de crédit
            if (prediction == 0 and proba > 61):
                pred, prob = st.columns(2)
                with prob:
                    st.plotly_chart(gauge,unsafe_allow_html=True)
                with pred:
                    st.markdown("<h2 style='text-align: center; color: #44be6e;'>AVIS FAVORABLE</h2>",
                                unsafe_allow_html=True)
            else:
                pred, prob = st.columns(2)
                with prob:
                    st.plotly_chart(gauge,unsafe_allow_html=True)
                with pred:
                    st.markdown("<h2 style='text-align: center; color: #ff3d41;'>AVIS DÉFAVORABLE</h2>",
                                unsafe_allow_html=True)
                # exemple de prêt refusé : ID=106854

        # Ligne de séparation :
        st.markdown("***")

        #Feature importance local
        show_local_feature_importance = st.checkbox(
            "Feature importance locale : données qui ont le plus influencé le calcul de ce score")
        if (show_local_feature_importance):
            shap.initjs()

            X = data[data['SK_ID_CURR'] == int(id_client)]
            X_pred_sans_id = X.drop(columns='SK_ID_CURR')
            X_pred = preprocessor.transform(X_pred_sans_id)
            fig, ax = plt.subplots(figsize=(15, 15))
            shap_values = explainer.shap_values(X_pred)
            st_shap(shap.force_plot(explainer_exp_value,shap_values, X_pred_sans_id))

        # Ligne de séparation :
        st.markdown("***")

    #
    # PARTIE informations descriptives de ce client
    #
        if (show_client_details):
            st.markdown("<h2 style='text-align: center; color: #5A5E6B;'>Informations descriptives relatives à ce client</h2>",
                        unsafe_allow_html=True)

            with st.spinner('Chargement des informations descriptives de ce client...'):

                age, rev_annuel, montant_pret, montant_annuite = st.columns(4)

                age.metric(label="Age", value=f"{abs(int(round(client_info.DAYS_BIRTH / 365, 25)))} ans")
                rev_annuel.metric(label="Revenus annuels", value=f"{int(round(client_info.AMT_INCOME_TOTAL))} $")
                montant_pret.metric(label="Montant du prêt", value=f"{int(round(client_info.AMT_CREDIT))} $")
                montant_annuite.metric(label="Montant des annuités", value=f"{int(round(client_info.AMT_ANNUITY))} $")


                # Ligne de démarcation :
                st.markdown("***")

    #
    # PARTIE Comparaison du client aux autres clients
    #
    var_comparaison = {
        'DAYS_BIRTH': "AGE",
        'AMT_INCOME_TOTAL': "REVENUS",
        'AMT_CREDIT': "MONTANT DU PRET",
        'AMT_ANNUITY': "MONTANT DES ANNUITES",
    }

    default_list = ["AGE", "REVENUS", "MONTANT DU PRET","MONTANT DES ANNUITES"]
    var_num = ['DAYS_BIRTH','AMT_INCOME_TOTAL', 'AMT_CREDIT','AMT_ANNUITY']


    if (show_client_comparison):
        st.header('Comparaison aux autres clients')

        with st.spinner('Chargement de la comparaison liée à la variable sélectionnée'):
            var = st.selectbox("Sélectionner une variable", \
                               list(var_comparaison.values()))
            variable = list(var_comparaison.keys()) \
                [list(var_comparaison.values()).index(var)]

            if (variable in var_num):
                graphique(data_info, variable, client_info[variable], var)
            else :
                st.write('tout est ok')


    #
    # PARTIE feature importance globale
    #

    if (shap_general):
        st.header('Feature importance globale')
        st.image('global_feature_importance.png')
else:
    st.markdown("**Identifiant non reconnu**")

#Lien streamlit dashboard
# https://laila770845-openclassroomsproject-dashboard-dashboard-dwus2u.streamlitapp.com/
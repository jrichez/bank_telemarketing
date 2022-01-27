# -*- coding: utf-8 -*-
"""
Created on Thu Jan 26 10:36:12 2022

@author: richez
"""

import streamlit as st
import pandas as pd
import numpy as np
from lightgbm import LGBMClassifier
import pickle

# Head

st.title('Campagne télémarketing banque')
st.markdown("Cette application repose sur un modèle basé sur les données d'une banque portugaise. Les données sont issues d'une campagne de marketing direct où les clients se sont vus proposés l'adhésion par contact téléphonique. Elle retourne la probabilité qu'une personne à contacter souscrive à l'offre de la banque. ")
st.markdown("Les données : https://www.kaggle.com/henriqueyamahata/bank-marketing")
st.markdown("Le modèle : https://www.kaggle.com/richez/bank-marketing/notebook")
st.subheader("2 possibilités pour obtenir les prédictions :")
st.markdown(" - Une prediction pour un prospect unique")
st.markdown(" - Un fichier client complété par une colonne contenant les probabilités")

# Middle

st.subheader('Fichier client')
st.markdown('Le fichier doit avoir la structure suivante :') 

# Exemple de fichier client
new_customer_1 = np.array([59, 'entrepreneur', 'married', 'professional.course', 'no', 'no', 'no', 'cellular', 'may', 'mon', 1, 999, 0, 'nonexistent', 1.1, 93.994, -36.4, 4.857, 5191.0])
new_customer_2 = np.array([29, 'unemployed', 'single', 'basic.4y', 'no', 'yes', 'yes', 'telephone', 'nov', 'fri', 1, 9, 1, 'success', -1.1, 94.767, -50.8, 1.028, 4963.6])
df_col = ['age', 'job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'day_of_week', 'campaign', 'pdays', 'previous', 'poutcome', 'emp.var.rate', 'cons.price.idx', 'cons.conf.idx', 'euribor3m', 'nr.employed']
df = pd.DataFrame([new_customer_1, new_customer_2], columns=df_col)
df_view = st.dataframe(df)


# Définir les variables du modèle
with st.expander("LES VARIABLES"):

	st.markdown("- 'Age' : âge de la personne contactée")

	st.markdown("- 'Job' : type de travail ")
	st.markdown("('admin.', 'blue-collar', 'entrepreneur', 'housemaid', 'management', 'retired', 'self-employed', 'services', 'student', 'technician', 'unemployed', 'unknown')")

	st.markdown("- 'Marital' : statut marital")
	st.markdown("('divorced', 'married', 'single', 'unknown')")

	st.markdown("- 'Education' : éducation ")
	st.markdown("('basic.4y', 'basic.6y', 'basic.9y', 'high.school', 'illiterate', 'professional.course', 'university.degree', 'unknown')")

	st.markdown("- 'Default' : a un crédit en défault de paiement ?")
	st.markdown("('Yes', 'No')")

	st.markdown("- 'Housing' : a un prêt immobilier ?")
	st.markdown("('Yes', 'No')")

	st.markdown("- 'Loan' : a un prêt personnel ?")
	st.markdown("('Yes', 'No')")

	st.markdown("- 'Contact' : type de contact")
	st.markdown("('cellular','telephone')")

	st.markdown("- 'Month' : mois du dernier contact")
	st.markdown("('jan', 'feb', '...', 'nov', 'dec')")

	st.markdown("- 'Dayofweek' : jour du dernier contact")
	st.markdown("'monday','tuesday','wednesday','thursday','friday')")

	st.markdown("- 'Campaign' : nombre de contacts")

	st.markdown("- 'Pdays' : nombre de jours passés entre la précédente campagne et un nouveau contact. Note : 999 indique que le client n'a pas été contacté lors de la précédente campagne")

	st.markdown("- 'Previous' : nombre de contacts réalisés avant cette campagne")

	st.markdown("- 'Poutcome' : résultat de la dernière campagne pour ce client s'il a été contacté")
	st.markdown("('failure','nonexistent','success')")

	st.markdown("- 'Emp.var.rate' : taux d'emploi")

	st.markdown("- 'Cons.price.idx' : indice des prix à la consommation")

	st.markdown("- 'Cons.conf.idx' : indice de confiance")

	st.markdown("- 'Euribor3m' : taux euribor 3 mois")


st.markdown('Le programme retourne :') 

df['Will not subscribe (%)'] = [0.2, 0.8]
df['Will subscribe (%)'] = [0.8, 0.2]
df_view = st.dataframe(df)

# Entrée du fichier client
customer_file = st.file_uploader('Importer un fichier client')

# Charger l'encodeur
encoder = pickle.load(open('encoder.pkl', 'rb'))

# Charger le scaler
scaler = pickle.load(open('scaler.pkl', 'rb'))

# Charger le modèle
model = pickle.load(open('model.sav', 'rb'))

var_cat_idx = [1, 2, 3, 4, 5, 6, 7, 8, 9, 13]
var_cat = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'day_of_week', 'poutcome']

@st.cache
def convert_df(df):
   return df.to_csv(index=False).encode('utf-8')

if customer_file is not None:
   customer_file = pd.read_csv(customer_file)
   df_to_predict = customer_file.copy()

   # Appliquer l'encodeur
   cat_enc = pd.DataFrame(encoder.transform(df_to_predict.iloc[:, var_cat_idx]).toarray())
   df_to_predict.drop(var_cat, axis=1, inplace=True)
   df_to_predict = pd.concat([df_to_predict, cat_enc], axis=1)

   # Appliquer le scaler
   df_to_predict = scaler.transform(df_to_predict)

   # Prédire les probas
   pred = model.predict_proba(df_to_predict)
   customer_file['Will not subscribe (%)'] = np.round(pred[:, 0], 2)
   customer_file['Will subscribe (%)'] = np.round(pred[:, 1], 2)

   # Convertir le dataframe en csv et le rendre téléchargeable
   df_csv = convert_df(customer_file)
   st.success('Les probabilités ont pu être prédites et le fichier complété est téléchargeable ci-dessous')
   st.download_button('Fichier client complété', df_csv, file_name='fichier_client.csv')

# Import du fichier test
test = pd.read_csv('test.csv')
test = convert_df(test)

# Note comprenant le fichier test
with st.expander('Note'):
	st.write("Un fichier est disponible pour pouvoir tester l'application : ")
	st.download_button('Fichier test', test, file_name='test.csv')

# Sidebar

title_pred = st.sidebar.title("Prospect unique")
st.sidebar.subheader("Sélectionner les différents attributs du prospect et le modèle retournera la probabilité qu'il souscrive à l'offre")

# Entrées des variables

with st.sidebar.form("my_form"):

	age = st.number_input('Age', step=1, value=52)
	job = st.selectbox('Profession', ('admin.', 'blue-collar', 'entrepreneur', 'housemaid', 'management', 'retired', 'self-employed', 'services', 'student', 'technician', 'unemployed', 'unknown'))
	marital = st.selectbox('Statut marital', ('divorced', 'married', 'single', 'unknown'))
	education = st.selectbox('Education', ('basic.4y', 'basic.6y', 'basic.9y', 'high.school', 'illiterate', 'professional.course', 'university.degree', 'unknown'))
	default = st.selectbox('Default de paiement', ('yes', 'no'))
	housing = st.selectbox('a un prêt immobilier ?', ('yes', 'no'))
	loan = st.selectbox('a un prêt personnel ?', ('yes', 'no'))
	contact = st.selectbox('type de contact ?', ('cellular','telephone'))
	month = st.selectbox(' mois du dernier contact', ('mar', 'apr', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec'))
	dow = st.selectbox(' jour du dernier contact', ('mon','tue','wed','thu','fri'))
	campaign = st.number_input('Nombre de contacts', step=1, value=3)
	pdays = st.number_input('Pdays', step=1, value=999)
	previous = st.number_input('Nombre de contacts avant campagne', step=1, value=6)
	poutcome = st.selectbox('Résultat de la dernière campagne', ('failure','nonexistent','success'))
	emp_var_rate = st.number_input("Taux d'emploi", step=0.1, value=1.1)
	cons_price_idx = st.number_input("Indice des prix à la consommation", step=0.1, value=93.9)
	cons_conf_idx = st.number_input("Indice de confiance", step=1, value=-36)
	euribor = st.number_input("Taux euribor 3 mois", step=0.1, value=4.8)
	nr_employed = st.number_input("Nombre d'employés", step=1, value=2)

	# Créer un array avec toutes les entrées
	new_pro = np.array([age, job, marital, education, default, housing, loan, contact, month, dow, campaign, pdays, previous, poutcome, emp_var_rate, cons_price_idx, cons_conf_idx, euribor, nr_employed])

	# Appliquer l'encodeur
	cat_endoded = encoder.transform(np.array(new_pro[var_cat_idx]).reshape(1, -1)).toarray()[0]
	new_pro = np.delete(new_pro, [var_cat_idx])
	new_pro = np.append(new_pro, cat_endoded)

	# Appliquer le scaler
	new_pro = scaler.transform(new_pro.reshape(1, -1))

	# Prédire les probas
	pred = model.predict_proba(new_pro.reshape(1, -1))[0]

	# Bouton de prédiction
	submitted = st.form_submit_button("Prédire")
	if submitted:
		col1, col2 = st.columns(2)
		col1.metric(label='Ne souscrira pas' , value=str(round(pred[0], 2)) + ' %')
		col2.metric(label='Souscrira' , value=str(round(pred[1], 2)) + ' %')






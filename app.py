# -*- coding: utf-8 -*-
"""
Created on Thu Jan 26 10:36:12 2022

@author: richez
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Head

st.title('Campagne télémarketing banque')
st.markdown("Cette application repose sur un modèle basé sur les données d'une banque portugaise. Les données sont issues d'une campagne de marketing direct où les clients se sont vus proposés l'adhésion par contact téléphonique. Elle retourne la probabilité qu'une personne à contacter souscrive à l'offre de la banque. ")
st.markdown("Les données : https://www.kaggle.com/henriqueyamahata/bank-marketing")
st.markdown("Le modèle : https://www.kaggle.com/richez/bank-marketing/notebook")
st.subheader("2 possibilités pour obtenir les prédictions :")
st.markdown(" - Une prediction pour un prospect unique")
st.markdown(" - Un fichier client complété par une colonne contenant les probabilités")

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


# Left

title_pred = st.sidebar.title("Prospect unique")
st.sidebar.subheader("Sélectionner les différents attributs du prospect et le modèle retournera la probabilité qu'il souscrive à l'offre")

# Intégrer les entrées des variables

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

	new_pro = np.array([age, job, marital, education, default, housing, loan, contact, month, dow, campaign, pdays, previous, poutcome, emp_var_rate, cons_price_idx, cons_conf_idx, euribor, nr_employed])

	encoder = pickle.load(open('encoder.pkl', 'rb'))
	var_cat_idx = [1, 2, 3, 4, 5, 6, 7, 8, 9, 13]
	cat_endoded = encoder.transform(np.array(new_pro[var_cat_idx]).reshape(1, -1)).toarray()[0]
	new_pro = np.delete(new_pro, [var_cat_idx])
	new_pro = np.append(new_pro, cat_endoded)

	scaler = pickle.load(open('scaler.pkl', 'rb'))
	new_pro = scaler.transform(new_pro.reshape(1, -1))

	model = pickle.load(open('model.sav', 'rb'))
	pred = model.predict_proba(new_pro.reshape(1, -1))[0]

	submitted = st.form_submit_button("Prédire")

	if submitted:
		col1, col2 = st.columns(2)
		col1.metric(label='Ne souscrira pas' , value=str(round(pred[0], 2)) + ' %')
		col2.metric(label='Souscrira' , value=str(round(pred[1], 2)) + ' %')
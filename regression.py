import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler,OneHotEncoder,LabelEncoder
import streamlit as st
import tensorflow as tf
import pickle as pkl


#load the trained model 
model = tf.keras.models.load_model('regression.keras')

#load the encoder and scalar 

with open('label_encoder_gender.pkl','rb') as file:
    label_encoder_gender = pkl.load(file)

with open ('onehot_encoder_geo.pkl','rb') as file:
    onehot_encoder_geo = pkl.load(file)

with open('scaler.pkl','rb') as file:
    scaler = pkl.load(file)

##streamlit 

st.title("Customer Churn Prediction")

# User input
geography = st.selectbox('Geography', onehot_encoder_geo.categories_[0])
gender = st.selectbox('Gender', label_encoder_gender.classes_)
age = st.slider('Age', 18, 92)
balance = st.number_input('Balance')
credit_score = st.number_input('Credit Score')
exited= st.selectbox('Exited', [0, 1])
tenure = st.slider('Tenure', 0, 10)
num_of_products = st.slider('Number of Products', 1, 4)
has_cr_card = st.selectbox('Has Credit Card', [0, 1])
is_active_member = st.selectbox('Is Active Member', [0, 1])

# Prepare the input data
input_data = pd.DataFrame({
    'CreditScore': [credit_score],
    'Gender': [label_encoder_gender.transform([gender])[0]],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active_member],
    'Exited': [exited]
})

#one-hot encode geography

geo_encoded = onehot_encoder_geo.transform([[geography]]).toarray()
geo_encoded_df = pd.DataFrame(geo_encoded,columns=onehot_encoder_geo.get_feature_names_out(['Geography']))

#combine one-hot encoded with input data 
input_data = pd.concat([input_data.reset_index(drop=True),geo_encoded_df],axis=1)

#scale the input data 

input_scaled = scaler.transform(input_data)

prediction = model.predict(input_scaled)


st.write((f'Estimated Salary: {prediction[0][0]:.2f}'))


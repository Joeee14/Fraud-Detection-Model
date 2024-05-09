import streamlit as st
import numpy as np
import joblib


logo_path = "Aiu.png"
st.sidebar.image(logo_path, caption='')
page=st.sidebar.selectbox("Explore or Predict",("Predict","Explore"))
model = joblib.load('KNN.pkl')
scaler = joblib.load('Scaler.pkl')

def show_Explore_Page():
    st.title('Fraud Detection System 💳')
    st.subheader('Data visualization 📊')
    matrix = "Matrix.png"
    st.image(matrix, caption='Confusion Matrix for the model')
def predict_fraud(features):
    features_scaled = scaler.transform(features)
    prediction = model.predict(features_scaled)
    return prediction[0]
def show_Predict_page():
    st.title('Fraud Detection System 💳')
    st.subheader("Enter required data for prediction 🔮")
    distance_from_home = st.number_input('Distance from Home', format="%.2f")
    distance_from_last_transaction = st.number_input('Distance from Last Transaction', format="%.2f")
    ratio_to_median_purchase_price = st.number_input('Ratio to Median Purchase Price', format="%.2f")
    repeat_retailer = st.selectbox('Repeat Retailer', options=[0, 1])
    used_chip = st.selectbox('Used Chip', options=[0, 1])
    used_pin_number = st.selectbox('Used PIN Number', options=[0, 1])
    online_order = st.selectbox('Online Order', options=[0, 1])
    if st.button('Predict Fraud'):
        features = np.array([[distance_from_home, distance_from_last_transaction, 
            ratio_to_median_purchase_price, repeat_retailer, used_chip, 
            used_pin_number, online_order]])
        result = predict_fraud(features)
        if result == 1:
            st.error('This transaction is likely a fraud! ⚠')
        else:
            st.success('This transaction is likely not a fraud.👍')
if page=="Predict":
    show_Predict_page()
else:
    show_Explore_Page()
    

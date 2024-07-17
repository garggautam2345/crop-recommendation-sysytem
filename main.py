import streamlit as st
import pandas as pd
import numpy as np
import helper

st.markdown(
    """
    <style>
    body {{
        background-image: url("https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSR7ahvb8aEN76vOIivqeFpa9_gBV5rZm2erw&s,{img_base64}");
        background-size: cover;
    }}
    .title {
        font-size: 50px;
        font-family: 'Arial Black', Gadget, sans-serif;
        color:#7719d4;
    }
    .subtitle {
        font-size: 30px;
        font-family: 'Courier New', Courier, monospace;
        color: #4682b4;
    }
    .custom-text {
        font-size: 20px;
        font-family: 'Comic Sans MS', cursive, sans-serif;
        color: #2e8b57;
    }
    </style>
    """,
    unsafe_allow_html=True
) 
df=pd.read_csv('Crop_recommendation.csv')
x_train,x_test,y_train,y_test=helper.x_and_y(df)
x_train_scalled,x_test_scalled=helper.scalling(x_train,x_test)
model=helper.model(x_train_scalled,y_train)
st.title("best crop prediction")
col1,col2,col3=st.columns(3)
with col1:
    N=st.text_input("nitrogen","133")
    temperature=st.text_input("temperature","24.40")    
with col2:
    P=st.text_input("Phosphorus","47")
    humidity=st.text_input("humidity","79.19")

with col3:
    K=st.text_input("Pottassium","24")
    ph=st.text_input("ph","7.21")

rainfall=st.text_input("rainfall","90.802236")


ans=helper.prediction(N,P,K,temperature,humidity,ph,rainfall,model)
st.title("based on the data you feed is best for your land ")
st.write(ans)
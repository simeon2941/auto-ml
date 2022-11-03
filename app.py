import streamlit as st
import pandas as pd
import os
import pandas_profiling  
from streamlit_pandas_profiling import st_profile_report
from operator import index
import plotly.express as px
from pycaret.regression import setup, compare_models, pull, save_model, load_model


with st.sidebar:
    st.image("https://images.pexels.com/photos/373543/pexels-photo-373543.jpeg?cs=srgb&dl=pexels-pixabay-373543.jpg&fm=jpg")
    st.title("AutML")
    choice = st.radio("Navitation",["Upload", "Profiling", "ML", "Download"])
    st.info("This app allows you to build an automated ML pipeline")

if os.path.exists("sourcedata.csv"):
    df = pd.read_csv("sourcedata.csv", index_col=None)

if choice == "Upload":
    st.title("Upload Data for Modeling!")
    file = st.file_uploader("Upload Your Dataset")
    if file:
        df = pd.read_csv(file, index_col=None)
        df.to_csv("sourcedata.csv", index=None)
        st.dataframe(df)
elif choice == "Profiling":
    st.title("Automated Data Analysis")
    profile_report = df.profile_report()
    st_profile_report(profile_report)
if choice == "Modelling": 
    chosen_target = st.selectbox('Choose the Target Column', df.columns)
    if st.button('Run Modelling'): 
        setup(df, target=chosen_target, silent=True)
        setup_df = pull()
        st.dataframe(setup_df)
        best_model = compare_models()
        compare_df = pull()
        st.dataframe(compare_df)
        save_model(best_model, 'best_model')

if choice == "Download": 
    with open('best_model.pkl', 'rb') as f: 
        st.download_button('Download Model', f, file_name="best_model.pkl")
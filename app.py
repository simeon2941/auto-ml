import streamlit as st
import pandas as pd
import os
import pandas_profiling  
from streamlit_pandas_profiling import st_profile_report

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
elif choice == "ML":
    pass
elif choice == "Download":
    pass
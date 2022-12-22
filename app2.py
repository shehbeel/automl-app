import streamlit as st
import os
import pandas as pd
import pandas_profiling
from streamlit_pandas_profiling import st_profile_report
from pycaret.classification import setup, compare_models, pull, save_model, load_model



st.title("[Auto]ML")
tab1, tab2, tab3, tab4, tab5 = st.tabs(["Upload", "Data Profiling", "Feature Selection", "Classification", "Clustering"])

if os.path.exists('./dataset.csv'): 
    df = pd.read_csv('dataset.csv', index_col=None)

with tab1:
    st.header("Upload Your Data")
    file = st.file_uploader("Upload Your Dataset")
    if file: 
        df = pd.read_csv(file, index_col=None)
        df.to_csv('dataset.csv', index=None)
        st.dataframe(df)
   
with tab2:
    st.header("Exploratory Data Analysis")
    # profile_df = df.profile_report()
    # st_profile_report(profile_df)

with tab3:
    st.header("Feature Selection")
    
with tab4:
    st.header("Classification")
    # chosen_target = st.selectbox('Choose the Target Column', df.columns)
    # if st.button('Run Modelling'): 
    #     setup(df, target=chosen_target, silent=True)
    #     setup_df = pull()
    #     st.dataframe(setup_df)
    #     best_model = compare_models()
    #     compare_df = pull()
    #     st.dataframe(compare_df)
    #     save_model(best_model, 'best_model')

with tab5:
    st.header("Clustering")



# with tab4:
#    st.header("A dog")
#    st.image("https://static.streamlit.io/examples/dog.jpg", width=200)

# with tab5:
#    st.header("An owl")
#    st.image("https://static.streamlit.io/examples/owl.jpg", width=200)





#streamlit run app2.py
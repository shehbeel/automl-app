import streamlit as st
import os
import pandas as pd
import pandas_profiling
from streamlit_pandas_profiling import st_profile_report
#from pycaret.classification import setup, compare_models, pull, save_model, load_model, plot_model, evaluate_model
from pycaret.classification import setup as class_setup
from pycaret.classification import compare_models as class_compare_models
from pycaret.classification import pull as class_pull
from pycaret.classification import save_model as class_save_model
from pycaret.classification import load_model as class_load_model
from pycaret.classification import plot_model as class_plot_model
from pycaret.classification import evaluate_model as class_evaluate_model
#from pycaret.clustering import *
from pycaret.clustering import setup as clust_setup
from pycaret.clustering import pull as clust_pull
from pycaret.clustering import create_model as clust_create_model
from pycaret.clustering import assign_model as clust_assign_model
from pycaret.clustering import plot_model as clust_plot_model


st.title("[Auto]ML")
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["Upload", "Data Profiling", "Feature Selection", "Classification", "Fine-Tune Model", "Clustering"])

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
    if st.button('Generate Report'):
        profile_df = df.profile_report()
        st_profile_report(profile_df)
    st.info('Warning! Only do this if dataset has small number of features.')

with tab3:
    st.header("Feature Selection")
    st.info('Coming Soon...Till then, enjoy this picture of an owl')
    st.image("https://static.streamlit.io/examples/owl.jpg", width=200)

    
with tab4: # Figure out 'fold' and how to make it selectable or as max as possible
    st.header("Classification")
    chosen_target = st.selectbox('Choose the Target Column:', df.columns)
    chosen_train_size = st.slider('Choose Size of Training Data:', min_value=0.0, max_value=1.0, step=0.01, value=0.7)
    chosen_kfold = st.slider('Choose Number of k-folds:', min_value=2, max_value=20, value=3)
    if st.button('Train Model'): 
        class_setup(df, target=chosen_target, silent=True, fold=chosen_kfold, train_size=chosen_train_size, preprocess=False, session_id=123) # ignore_low_variance = True, normalize=True
        setup_df = class_pull()
        st.info('This is the ML experiment settings')
        st.dataframe(setup_df)
        best_model = class_compare_models()
        compare_df = class_pull()
        st.info("This is the ML model")
        st.dataframe(compare_df)
        #best_model # To see the model parameters of best model
        class_save_model(best_model, 'best_model')
        #st.download_button()
        
        # Plot Best Features
        # st.info('Classification Report')
        # ##feature_plot = plot_model(best_model, plot = 'feature', display_format='streamlit')
        # class_report = class_plot_model(best_model, plot = 'class_report', display_format='streamlit')
        # auc_plot = class_plot_model(best_model, plot = 'auc', display_format='streamlit')
        # conf_matrix = class_plot_model(best_model, plot = 'confusion_matrix', display_format='streamlit') #plot_kwargs = {'percent' : True}
        # error_plot = class_plot_model(best_model, plot = 'error', display_format='streamlit')   
        # ##bound_plot = plot_model(best_model, plot = 'boundary', use_train_data = True, display_format='streamlit')
        # feature_plot = class_plot_model(best_model, plot = 'feature', use_train_data = True, display_format='streamlit')
        # threshold_plot = class_plot_model(best_model, plot = 'threshold', display_format='streamlit')
        # ##rfe_plot = plot_model(best_model, plot = 'rfe', use_train_data = True, display_format='streamlit')
        # pr_plot = class_plot_model(best_model, plot = 'pr', display_format='streamlit')
        # learning_curve = class_plot_model(best_model, plot = 'learning', display_format='streamlit')
        # val_curve = class_plot_model(best_model, plot = 'vc', display_format='streamlit')

        # st.set_option('deprecation.showPyplotGlobalUse', False)
        # st.pyplot(auc_plot)
        # st.pyplot(class_report)
        # st.pyplot(conf_matrix)
        # st.pyplot(error_plot)
        # st.pyplot(feature_plot)
        # st.pyplot( threshold_plot)
        # st.pyplot(pr_plot)
        # st.pyplot(learning_curve)
        # st.pyplot(val_curve)

with tab5:
    st.header("Fine-Tune Model")
    st.info('Coming Soon...Till then, enjoy this picture of a dog')
    st.image("https://static.streamlit.io/examples/dog.jpg", width=200)

with tab6:
    st.header("Clustering")
    selected_cols = st.multiselect('Select features to remove for clustering', df.columns)
    chosen_clust_model = st.selectbox('This is the clustering settings', ('kmeans', 'ap', 'meanshift', 'sc', 'hclust', 'dbscan', 'optics', 'birch', 'kmodes'))
    chosen_num_clusters = st.number_input('Select number of clusters, k', min_value=1, step=1)
    if st.button('Start Clustering!'): 
        clust_setup(df, normalize=True, ignore_features=selected_cols, silent=True, session_id=123)
        clust_setup_df = clust_pull()
        st.info('This is the clustering settings')
        st.dataframe(clust_setup_df)
        clust_model = clust_create_model(chosen_clust_model, num_clusters=chosen_num_clusters)
        model_results = clust_assign_model(clust_model)
        model_results_df = clust_pull()
        st.info("Clustering Results:")
        st.dataframe(model_results_df)  
        st.set_option('deprecation.showPyplotGlobalUse', False)
        # PCA Plot
        pca_plot = clust_plot_model(clust_model, display_format='streamlit')   
        st.pyplot(pca_plot)
        # Elbow Plot
        elbow_plot = clust_plot_model(clust_model, plot = 'elbow', display_format='streamlit')   
        st.pyplot(elbow_plot)
        # Silhouette Plot
        silhouette_plot = clust_plot_model(clust_model, plot = 'silhouette', display_format='streamlit')   
        st.pyplot(silhouette_plot)
        # Distribution Plot
        distribution_plot = clust_plot_model(clust_model, plot = 'distribution', display_format='streamlit') #to see size of clusters  
        st.pyplot(distribution_plot)


        



#streamlit run app.py

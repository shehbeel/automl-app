import streamlit as st
import os
import pandas as pd
import pandas_profiling
from streamlit_pandas_profiling import st_profile_report
import pickle
#from pycaret.classification import setup, compare_models, pull, save_model, load_model, plot_model, evaluate_model
from pycaret.classification import setup as class_setup
from pycaret.classification import compare_models as class_compare_models
from pycaret.classification import pull as class_pull
from pycaret.classification import save_model as class_save_model
from pycaret.classification import load_model as class_load_model
from pycaret.classification import plot_model as class_plot_model
from pycaret.classification import evaluate_model as class_evaluate_model
from pycaret.classification import create_model as class_create_model
from pycaret.classification import tune_model as class_tune_model
from pycaret.classification import finalize_model as class_finalize_model
#from pycaret.classification import predict_model as class_predict_model
#from pycaret.classification import get_leaderboard as class_get_leaderboard
#from pycaret.classification import interpret_model as class_interpret_model

#from pycaret.classification import models as class_models

#from pycaret.clustering import *
from pycaret.clustering import setup as clust_setup
from pycaret.clustering import pull as clust_pull
from pycaret.clustering import create_model as clust_create_model
from pycaret.clustering import assign_model as clust_assign_model
from pycaret.clustering import plot_model as clust_plot_model


st.title("[Auto]ML")
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["Upload", "Data Profiling", "Data Preprocessing", "Classification", "Fine-Tune Model", "Clustering"])

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
    chosen_target = st.selectbox('Choose the Target Column:', df.columns)
    chosen_train_size = st.slider('Choose Size of Training Data:', min_value=0.0, max_value=1.0, step=0.01, value=0.7)
    stratify_result = st.checkbox('Stratify Training/Testing Data')
    fold_strat_result = st.selectbox('Choose Cross-Validation Method:', ['kfold','stratifiedkfold','groupkfold'])
    chosen_kfold = st.slider('Choose Number of k-folds:', min_value=2, max_value=10, value=5)
    st.subheader('Data Preprocessing')
    normalize_result = st.checkbox('Normalize the Data')
    normalization_method_result = st.selectbox('Select normalization method:', ['zscore','minmax','maxabs','robust'])
    transform_result = st.checkbox('Transform the Data')
    transform_method_result = st.selectbox('Select Data Transformation Method:',['yeo-johnson','quantile'])
    imputation_type_result = st.selectbox('Select imputation method for missing data.', ['mean','median','zero'])
    ilv_result = st.checkbox('Ignore Low Variance Features')
    ignore_features_result = st.multiselect('Select features to ignore.', df.columns)
    remove_outliers_result = st.checkbox('Remove Outlier Samples')
    outliers_threshold_result = st.slider('Select the precentage outliers to be removed:', min_value=0.0, max_value=1.0, value=0.05)
    feature_select_result = st.checkbox('Perform Feature Selection')
    feature_select_threshold_result = st.slider('Select Feature Selection Threshold:', min_value=0.0, max_value=1.0, value=0.8)
    feature_select_method_result = st.selectbox('Select Feature Selection Method.', ['classic','boruta'])
    fix_imbalance_result = st.checkbox('Fix Target Class Imbalance')
    if st.button('Preprocess the Data'): 
        class_setup(df, target=chosen_target, silent=True, 
                    data_split_stratify=stratify_result,
                    fold_strategy=fold_strat_result,
                    fold=chosen_kfold, 
                    train_size=chosen_train_size,
                    preprocess=False,
                    normalize=normalize_result,
                    normalize_method=normalization_method_result,
                    transformation=transform_result,
                    transformation_method=transform_method_result,
                    numeric_imputation=imputation_type_result,
                    ignore_low_variance = ilv_result,
                    ignore_features=ignore_features_result,
                    remove_outliers=remove_outliers_result,
                    outliers_threshold=outliers_threshold_result,
                    feature_selection=feature_select_result,
                    feature_selection_threshold=feature_select_threshold_result, 
                    feature_selection_method=feature_select_method_result,        
                    fix_imbalance=fix_imbalance_result, 
                    profile=True, 
                    session_id=123)
        setup_df = class_pull()
        st.subheader('Experiment Settings')
        st.dataframe(setup_df)

    
with tab4: # Figure out 'fold' and how to make it selectable or as max as possible
    st.header("Classification")
    # chosen_target = st.selectbox('Choose the Target Column:', df.columns)
    # chosen_train_size = st.slider('Choose Size of Training Data:', min_value=0.0, max_value=1.0, step=0.01, value=0.7)
    # chosen_kfold = st.slider('Choose Number of k-folds:', min_value=2, max_value=10, value=5)
    if st.button('Train Model'): 
    #     class_setup(df, target=chosen_target, silent=True, 
    #                 fold=chosen_kfold, 
    #                 train_size=chosen_train_size,
    #                 preprocess=False,
    #                 # normalize=True,
    #                 # fix_imbalance=True, 
    #                 # feature_selection=True,
    #                 # feature_selection_threshold=0.8,
    #                 # ignore_low_variance = True,
    #                 # profile=True, 
    #                 session_id=123)
    #     setup_df = class_pull()
    #     st.subheader('Experiment Settings')
    #     st.dataframe(setup_df)
        best_model = class_compare_models()
        compare_df = class_pull()
        st.subheader("Machine Learning Models")
        st.dataframe(compare_df)
        st.caption("These are the parameters for the best ML model:")
        best_model # To see the model parameters of best model
        #class_save_model(best_model, 'best_model')
        #Finalize model:
        final_best_model = class_finalize_model(best_model)
        st.download_button("Download Best Model",
                            data=pickle.dumps(final_best_model),
                            file_name="model.pkl",
                            )

with tab5:
    st.header("Fine-Tune Model")
    # Show table of all models available
    # st.subheader('Models available:')
    # all_models = class_models()
    # all_models_df = class_pull()
    pycaret_models = ['lr','knn','nb','dt','svm','rbfsvm','gpc','mlp','ridge','rf','qda','ada','gbc','lda','et','xgboost','lightgbm','catboost']
    chosen_n_iters = st.slider('Choose number of iterations for hyperparameter tuning. The higher the iterations the better the model optimization, but will require more time.', min_value=1, max_value=100, step=1, value=10)
    chosen_model = st.selectbox('Select an ML model', pycaret_models)
    if st.button('Tune Model'):
        # class_setup(df, target=chosen_target, silent=True, 
        #             fold=chosen_kfold, 
        #             train_size=chosen_train_size,
        #             preprocess=False,
        #             # normalize=True,
        #             # fix_imbalance=True, 
        #             # feature_selection=True,
        #             # feature_selection_threshold=0.8,
        #             # ignore_low_variance = True,
        #             # profile=True, 
        #             session_id=123)
        # setup_df = class_pull()
        # st.subheader('Experiment Settings')
        # st.dataframe(setup_df)
        chosen_mdl = class_create_model(chosen_model)
        # sklearn Random Grid Search is used (https://pycaret.readthedocs.io/en/stable/api/classification.html#)
        tuned_model, tuner = class_tune_model(chosen_mdl, n_iter=chosen_n_iters, choose_better = True, early_stopping=True, return_tuner=True) # n_iter=50, search_library = 'tune-sklearn', search_algorithm = 'hyperopt'
        tuned_model_df = class_pull()
        st.subheader('Hyperparameter Tuning Results')
        st.dataframe(tuned_model_df)
        st.caption("These are the parameters for the best tuned ML model. Note: If hyperparameter tuning did not help, then base model will be the final model selected.")
        tuned_model
        st.caption("These are the parameters for the model tuner:")
        tuner
        #Finalize model:
        final_tuned_model = class_finalize_model(tuned_model)
        
        # Download model:
        st.download_button("Download Optimized Model",
                            data=pickle.dumps(final_tuned_model),
                            file_name="optimized-model.pkl",
                            )        

        # tuned_model_pred = class_predict_model(tuned_model, data=df)
        # tuned_model_pred_df = class_pull()
        # st.dataframe(tuned_model_pred_df)
        # class_evaluate_model(tuned_model)
        # class_interpret_model(tuned_model)


with tab6:
    st.header("Clustering")
    selected_cols = st.multiselect('Select features to remove for clustering', df.columns)
    chosen_clust_model = st.selectbox('This is the clustering settings', ('kmeans', 'ap', 'meanshift', 'sc', 'hclust', 'dbscan', 'optics', 'birch', 'kmodes'))
    chosen_num_clusters = st.number_input('Select number of clusters, k', min_value=1, step=1)
    if st.button('Start Clustering!'): 
        clust_setup(df, normalize=True, ignore_features=selected_cols, silent=True, session_id=123)
        clust_setup_df = clust_pull()
        st.caption('This is the clustering settings')
        st.dataframe(clust_setup_df)
        clust_model = clust_create_model(chosen_clust_model, num_clusters=chosen_num_clusters)
        model_results = clust_assign_model(clust_model)
        model_results_df = clust_pull()
        st.caption("Clustering Results:")
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


        
# st.info('Coming Soon...Till then, enjoy this picture of an owl')
# st.image("https://static.streamlit.io/examples/owl.jpg", width=200)

# st.info('Coming Soon...Till then, enjoy this picture of a dog')
# st.image("https://static.streamlit.io/examples/dog.jpg", width=200)
#streamlit run app.py

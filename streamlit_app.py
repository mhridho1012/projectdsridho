import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import altair as alt
import time
import zipfile

# Page title
st.set_page_config(page_title='ML model builder', page_icon='🏗️')
st.title('Prediksi Nilai SPPBK menggunakan Machine Learning')

# Adding background image
background_url = "https://media.istockphoto.com/id/2124093009/id/foto/palm-oil-and-oil-palm.jpg?s=2048x2048&w=is&k=20&c=sY6mhX8YvdSO7Hwe-kd0OBGL7UiyzIkUyh3z6YXTKXQ="  # Replace with your Unsplash image URL
st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url("{background_url}");
        background-size: cover;
    }}
    .box {{
        background: rgba(255, 255, 255, 0.8);
        padding: 15px;
        border-radius: 10px;
        margin-bottom: 20px;
    }}
    </style>
    """,
    unsafe_allow_html=True
)

# Function to create a box with content
def create_box(content):
    return st.markdown(f'<div class="box">{content}</div>', unsafe_allow_html=True)

with st.expander('About this app'):
    create_box('**What can this app do?**')
    st.info('This app allows users to build a machine learning (ML) model in an end-to-end workflow. Particularly, this encompasses data upload, data pre-processing, ML model building and post-model analysis.')

    create_box('**How to use the app?**')
    st.warning('To engage with the app, go to the sidebar and 1. Select a data set and 2. Adjust the model parameters by adjusting the various slider widgets. As a result, this would initiate the ML model building process, display the model results as well as allowing users to download the generated models and accompanying data.')

    create_box('**Under the hood**')
    st.markdown('Data sets:')
    st.code('''- Drug solubility data set
    ''', language='markdown')

    create_box('Libraries used:')
    st.code('''- Pandas for data wrangling
- Scikit-learn for building a machine learning model
- Altair for chart creation
- Streamlit for user interface
    ''', language='markdown')

# Sidebar for accepting input parameters
with st.sidebar:
    # Load data
    st.header('1.1. Input data')

    create_box('**1. Use custom data**')
    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file, index_col=False)

    # Download example data
    @st.cache_data
    def convert_df(input_df):
        return input_df.to_csv(index=False).encode('utf-8')
    example_csv = pd.read_csv('https://raw.githubusercontent.com/dataprofessor/data/master/delaney_solubility_with_descriptors.csv')
    csv = convert_df(example_csv)
    st.download_button(
        label="Download example CSV",
        data=csv,
        file_name='delaney_solubility_with_descriptors.csv',
        mime='text/csv',
    )

    create_box('**1.2. Use example data**')
    example_data = st.toggle('Load example data')
    if example_data:
        df = pd.read_csv('https://raw.githubusercontent.com/dataprofessor/data/master/delaney_solubility_with_descriptors.csv')

    st.header('2. Set Parameters')
    parameter_split_size = st.slider('Data split ratio (% for Training Set)', 10, 90, 80, 5)

    st.subheader('2.1. Learning Parameters')
    with st.expander('See parameters'):
        parameter_n_estimators = st.slider('Number of estimators (n_estimators)', 0, 1000, 100, 100)
        parameter_max_features = st.select_slider('Max features (max_features)', options=['all', 'sqrt', 'log2'])
        parameter_min_samples_split = st.slider('Minimum number of samples required to split an internal node (min_samples_split)', 2, 10, 2, 1)
        parameter_min_samples_leaf = st.slider('Minimum number of samples required to be at a leaf node (min_samples_leaf)', 1, 10, 2, 1)

    st.subheader('2.2. General Parameters')
    with st.expander('See parameters', expanded=False):
        parameter_random_state = st.slider('Seed number (random_state)', 0, 1000, 42, 1)
        parameter_criterion = st.select_slider('Performance measure (criterion)', options=['squared_error', 'absolute_error', 'friedman_mse'])
        parameter_bootstrap = st.select_slider('Bootstrap samples when building trees (bootstrap)', options=[True, False])
        parameter_oob_score = st.select_slider('Whether to use out-of-bag samples to estimate the R^2 on unseen data (oob_score)', options=[False, True])

    sleep_time = st.slider('Sleep time', 0, 3, 0)

# Initiate the model building process
if uploaded_file or example_data:
    with st.status("Running ...", expanded=True) as status:

        st.write("Loading data ...")
        time.sleep(sleep_time)

        st.write("Preparing data ...")
        time.sleep(sleep_time)
        X = df.iloc[:, :-1]
        y = df.iloc[:, -1]

        st.write("Splitting data ...")
        time.sleep(sleep_time)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=(100 - parameter_split_size) / 100, random_state=parameter_random_state)

        st.write("Model training ...")
        time.sleep(sleep_time)

        if parameter_max_features == 'all':
            parameter_max_features = None
            parameter_max_features_metric = X.shape[1]

        rf = RandomForestRegressor(
            n_estimators=parameter_n_estimators,
            max_features=parameter_max_features,
            min_samples_split=parameter_min_samples_split,
            min_samples_leaf=parameter_min_samples_leaf,
            random_state=parameter_random_state,
            criterion=parameter_criterion,
            bootstrap=parameter_bootstrap,
            oob_score=parameter_oob_score)
        rf.fit(X_train, y_train)

        st.write("Applying model to make predictions ...")
        time.sleep(sleep_time)
        y_train_pred = rf.predict(X_train)
        y_test_pred = rf.predict(X_test)

        st.write("Evaluating performance metrics ...")
        time.sleep(sleep_time)
        train_mse = mean_squared_error(y_train, y_train_pred)
        train_r2 = r2_score(y_train, y_train_pred)
        test_mse = mean_squared_error(y_test, y_test_pred)
        test_r2 = r2_score(y_test, y_test_pred)

        st.write("Displaying performance metrics ...")
        time.sleep(sleep_time)
        parameter_criterion_string = ' '.join([x.capitalize() for x in parameter_criterion.split('_')])
        rf_results = pd.DataFrame(['Random forest', train_mse, train_r2, test_mse, test_r2]).transpose()
        rf_results.columns = ['Method', f'Training {parameter_criterion_string}', 'Training R2', f'Test {parameter_criterion_string}', 'Test R2']
        # Convert objects to numerics
        for col in rf_results.columns:
            rf_results[col] = pd.to_numeric(rf_results[col], errors='ignore')
        # Round to 3 digits
        rf_results = rf_results.round(3)

    status.update(label="Status", state="complete", expanded=False)

    # Display data info
    st.header('Input data', divider='rainbow')
    col = st.columns(4)
    col[0].metric(label="No. of samples", value=X.shape[0], delta="")
    col[1].metric(label="No. of X variables", value=X.shape[1], delta="")
    col[2].metric(label="No. of Training samples", value=X_train.shape[0], delta="")
    col[3].metric(label="No. of Test samples", value=X_test.shape[0], delta="")

    with st.expander('Initial dataset', expanded=True):
        create_box(st.dataframe(df, height=210, use_container_width=True))
    with st.expander('Train split', expanded=False):
        train_col = st.columns((3, 1))
        with train_col[0]:
            st.markdown('**X**')
            create_box(st.dataframe(X_train, height=210, hide_index=True, use_container_width=True))
        with train_col[1]:
            st.markdown('**y**')
            create_box(st.dataframe(y_train, height=210, hide_index=True, use_container_width=True))
    with st.expander('Test split', expanded=False):
        test_col = st.columns((3, 1))
        with test_col[0]:
            st.markdown('**X**')
            create_box(st.dataframe(X_test, height=210, hide_index=True, use_container_width=True))
        with test_col[1]:
            st.markdown('**y**')
            create_box(st.dataframe(y_test, height=210, hide_index=True, use_container_width=True))

    st.header('Results', divider='rainbow')
    st.subheader('Performance metrics')
    st.dataframe(rf_results, use_container_width=True)

    st.subheader('Prediction plots')
    with st.expander('See plots'):
        fig_col = st.columns(2)
        with fig_col[0]:
            st.markdown('**Train set**')
            rf_train = pd.DataFrame({'Actual': y_train, 'Predicted': y_train_pred})
            rf_train['Residual'] = rf_train['Actual'] - rf_train['Predicted']
            rf_train = rf_train.reset_index(drop=True).reset_index()
            fig = alt.Chart(rf_train).transform_fold(
                fold=['Actual', 'Predicted'],
                as_=['category', 'value']
            ).mark_line().encode(
                x='index:Q',
                y='value:Q',
                color='category:N'
            ).properties(
                height=200
            ).interactive()
            st.altair_chart(fig, use_container_width=True)
            st.markdown('**Test set**')
            rf_test = pd.DataFrame({'Actual': y_test, 'Predicted': y_test_pred})
            rf_test['Residual'] = rf_test['Actual'] - rf_test['Predicted']
            rf_test = rf_test.reset_index(drop=True).reset_index()
            fig = alt.Chart(rf_test).transform_fold(
                fold=['Actual', 'Predicted'],
                as_=['category', 'value']
            ).mark_line().encode(
                x='index:Q',
                y='value:Q',
                color='category:N'
            ).properties(
                height=200
            ).interactive()
            st.altair_chart(fig, use_container_width=True)
        with fig_col[1]:
            st.markdown('**Residuals**')
            fig = alt.Chart(rf_train).transform_density(
                density='Residual',
                groupby=['category'],
                as_=['Residual', 'density']
            ).mark_area(opacity=0.3).encode(
                x='Residual:Q',
                y='density:Q',
                color='category:N'
            ).properties(
                height=200
            ).interactive()
            st.altair_chart(fig, use_container_width=True)
            create_box(st.write(rf_train.describe()))

    # Download model and data
    st.header('Export', divider='rainbow')
    create_box('**Dataframes and model can be downloaded in a zip file**')
    dfs = {'df': df, 'X_train': X_train, 'X_test': X_test, 'y_train': y_train, 'y_test': y_test, 'rf_results': rf_results}
    dfs_path = '/mnt/data/dfs.zip'
    with zipfile.ZipFile(dfs_path, 'w') as zf:
        for name, df in dfs.items():
            df.to_csv(f'{name}.csv', index=False)
            zf.write(f'{name}.csv')
    st.download_button(
        label="Download ZIP",
        data=dfs_path,
        file_name='dfs.zip',
    )
else:
    st.warning('No data available. Please upload a CSV file or use the example data to proceed.')

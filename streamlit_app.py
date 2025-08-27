import streamlit as st
import joblib
import pandas as pd
import numpy as np
import os

# --- Custom CSS ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;700&display=swap');
    
    html, body, [class*="st-"] {
        font-family: 'Roboto', sans-serif;
    }

    /* Main title styling */
    .st-emotion-cache-183n00d {
        color: #FFFFFF;
        background: -webkit-linear-gradient(45deg, #1A5276, #A9CCE3);
        background-clip: text;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 700;
        text-shadow: 2px 2px 5px rgba(0,0,0,0.1);
    }
    
    /* Subtitle styling */
    h3 {
        color: #FFFFFF;
    }

    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background-color: #2E4053;
        color: #EAECEE;
    }
    
    /* Selectbox styling */
    .st-emotion-cache-n22h0a {
        border-color: #5DADE2;
    }
    .st-emotion-cache-n22h0a:hover {
        border-color: #4CAF50;
    }
    
    /* Button styling */
    .st-emotion-cache-7ym5gk {
        background-color: #5DADE2;
        color: #FFFFFF;
        border: none;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.2);
    }
    .st-emotion-cache-7ym5gk:hover {
        background-color: #4CAF50;
    }
    
    /* Success box styling */
    [data-testid="stSuccess"] {
        background-color: #A9CCE3;
        color: #1A5276;
        border-radius: 10px;
        border: 2px solid #5DADE2;
        box-shadow: 4px 4px 10px rgba(0,0,0,0.1);
    }
    
    /* Data table styling */
    .st-emotion-cache-90vs1b {
        border-radius: 10px;
        overflow: hidden;
    }
</style>
""", unsafe_allow_html=True)

# --- Page Configuration ---
st.set_page_config(
    page_title="Product Recommender",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'About': "This is a custom-built e-commerce recommendation app."
    }
)
# Use Streamlit's caching to load the model and data only once
# This is crucial for performance as it avoids re-loading on every user interaction
@st.cache_resource
def load_resources():
    """
    Loads the trained model and unique items data for the application.
    Returns:
        tuple: (trained_model, all_items_df, training_columns)
    """
    try:
        if not os.path.exists('final_xgb_model.pkl'):
            st.error("Error: 'final_xgb_model.pkl' not found. Please ensure it's in the app's directory.")
            st.stop()
        if not os.path.exists('sample_df.csv'):
            st.error("Error: 'sample_df.csv' not found. Please ensure it's in the app's directory.")
            st.stop()
        if not os.path.exists('training_columns.txt'):
            st.warning("Warning: 'training_columns.txt' not found. Using feature names from the model. For production, it is recommended to save and load the training columns explicitly.")

        final_xgb_model = joblib.load('final_xgb_model.pkl')
        all_items_df = pd.read_csv('sample_df.csv')
        
        if os.path.exists('training_columns.txt'):
            with open('training_columns.txt', 'r') as f:
                training_columns = [line.strip() for line in f]
        else:
            # Fallback method: If the file doesn't exist, get feature names from the model
            training_columns = final_xgb_model.feature_names
            st.warning("Could not find 'training_columns.txt'. Using feature names from the model. "
                       "For production, it is recommended to save and load the training columns explicitly.")
            
        return final_xgb_model, all_items_df, training_columns
    except FileNotFoundError as e:
        st.error(f"Error loading files: {e}. Please make sure 'final_xgb_model.pkl' and 'unique_items.csv' are in the same directory.")
        st.stop()
    except AttributeError:
        st.error("The loaded model does not have 'feature_names'. This may happen if the model was not trained with a dataframe. Please re-train and save your model correctly.")
        st.stop()

final_xgb_model, all_items_df, training_columns = load_resources()

# --- Recommendation Function ---
def recommend_items_for_user(visitorid, all_items_df, trained_model, training_columns, top_n=5):
    """
    Generates a list of top-N recommended items for a given user based on the model's predictions.
    
    Args:
        visitorid (int): The ID of the visitor to generate recommendations for.
        all_items_df (pd.DataFrame): The DataFrame containing all unique items and their properties.
        trained_model (XGBClassifier): The trained XGBoost model.
        training_columns (list): The list of feature columns the model was trained on.
        top_n (int): The number of top recommendations to return.

    Returns:
        pd.DataFrame: A DataFrame of the top-N recommended items with their predicted likelihood.
    """
    items_to_predict = all_items_df.copy()
    items_to_predict['visitorid'] = visitorid
    
    items_to_predict_encoded = pd.get_dummies(items_to_predict, columns=['categoryid', 'parentid'], prefix=['cat', 'parent'])
    
    items_to_predict_encoded = items_to_predict_encoded.reindex(columns=training_columns, fill_value=0)
    
    if not all(col in items_to_predict_encoded.columns for col in training_columns):
        missing_cols = set(training_columns) - set(items_to_predict_encoded.columns)
        st.error(f"Prediction failed: The DataFrame is missing critical columns: {list(missing_cols)}")
        st.stop()
    
    probabilities = trained_model.predict_proba(items_to_predict_encoded)[:, 1]
    
    items_to_predict['likelihood'] = probabilities

    top_recommendations = items_to_predict.sort_values(by='likelihood', ascending=False).head(top_n)
    
    return top_recommendations[['itemid', 'categoryid', 'likelihood']]


# --- Streamlit UI ---
st.markdown("<h1 style='text-align: center; color: #4CAF50;'>🛍️ E-Commerce Recommendation Engine 🛍️</h1>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align: center; color: #66BB6A;'>Powered by XGBoost</h3>", unsafe_allow_html=True)

st.write("This application demonstrates the predictive power of our machine learning recommendation model. Select a visitor ID from the sidebar and get instant, personalized recommendations!")

# Sidebar for user input
with st.sidebar:
    st.header("Visitor Selection")
    # Get a list of unique visitor IDs to populate the selectbox
    if 'visitorid' in all_items_df.columns:
        unique_visitors = all_items_df['visitorid'].unique()
        visitor_id = st.selectbox(
            'Select a Visitor ID:',
            unique_visitors
        )

        if st.button('Get Recommendations'):
            if visitor_id:
                with st.spinner('Generating recommendations...'):
                    recommendations = recommend_items_for_user(
                        visitor_id,
                        all_items_df,
                        final_xgb_model,
                        training_columns,
                        top_n=5
                    )
                    st.session_state.recommendations = recommendations
                    st.session_state.visitor_id = visitor_id
            else:
                st.warning('Please select a Visitor ID.')
    else:
        st.error("The 'visitorid' column was not found in the unique_items.csv file. Please check your data.")

# Main content area for displaying results
st.markdown("---")
if 'recommendations' in st.session_state and st.session_state.recommendations is not None:
    st.success(f"Top 10 Recommendations for Visitor {st.session_state.visitor_id}:")
    
    # Display the table with a bit of styling
    st.table(st.session_state.recommendations.style.format({'likelihood': '{:.2f}'}))
else:
    st.info("Select a visitor ID from the sidebar and click 'Get Recommendations' to see results.")

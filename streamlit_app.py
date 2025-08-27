import streamlit as st
import joblib
import pandas as pd
import numpy as np
import os

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
st.title('E-Commerce Recommendation Engine')

st.write("This application demonstrates the predictive power of our XGBoost recommendation model.")

# Get a list of unique visitor IDs to populate the selectbox
unique_visitors = all_items_df['visitorid'].unique()
visitor_id = st.selectbox(
    'Select a Visitor ID to get recommendations:',
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
            
            st.success(f"Top 5 Recommendations for Visitor {visitor_id}:")
            st.dataframe(recommendations.style.format({'likelihood': '{:.2f}'}), use_container_width=True)
    else:
        st.warning('Please select a Visitor ID.')

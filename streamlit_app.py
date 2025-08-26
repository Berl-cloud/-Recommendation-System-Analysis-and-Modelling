import streamlit as st
import joblib
import pandas as pd
import numpy as np

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
        final_xgb_model = joblib.load('final_xgb_model.pkl')
        all_items_df = pd.read_csv('sample_df.csv')
        
        training_columns = final_xgb_model.feature_names
        return final_xgb_model, all_items_df, training_columns
    except FileNotFoundError as e:
        st.error(f"Error loading files: {e}. Please make sure 'final_xgb_model.pkl' and 'unique_items.csv' are in the same directory.")
        st.stop()

final_xgb_model, all_items_df, training_columns = load_resources()

# --- Recommendation Function ---
def recommend_items_for_user(visitorid, all_items_df, trained_model, training_columns, top_n=5):
    """
    Generates a list of top-N recommended items for a given user based on the model's predictions.
    """
    items_to_predict = all_items_df.copy()
    items_to_predict['visitorid'] = visitorid
    
    items_to_predict_encoded = pd.get_dummies(items_to_predict, columns=['categoryid', 'parentid'], prefix=['cat', 'parent'])
    
    items_to_predict_encoded = items_to_predict_encoded.reindex(columns=training_columns, fill_value=0)
    

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
                top_n=10
            )
            
            st.success(f"Top 10 Recommendations for Visitor {visitor_id}:")
            st.dataframe(recommendations.style.format({'likelihood': '{:.2f}'}), use_container_width=True)
    else:
        st.warning('Please select a Visitor ID.')

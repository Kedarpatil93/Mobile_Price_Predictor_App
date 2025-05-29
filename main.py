import streamlit as st
import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
import category_encoders as ce

# Define the custom transformer class BEFORE loading the pipeline
class TargetEncoderWrapper(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.encoder = ce.TargetEncoder()

    def fit(self, X, y):
        self.encoder.fit(X, y)
        return self

    def transform(self, X):
        return self.encoder.transform(X)

# load the pipeline and data using joblib
pipe = joblib.load('pipe.joblib')
df = joblib.load('df.joblib')

# Page configuration
st.set_page_config(page_title="Mobile Price Predictor", layout="centered")
st.markdown("<h1 style='text-align: center; color: #4B8BBE;'>📱 Mobile Price Predictor</h1>", unsafe_allow_html=True)

# UI Styling
st.markdown("""
    <style>
        .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
        }
        h3 {
            color: #4B8BBE;
        }
        .stRadio > label {
            font-weight: bold;
        }
        .stSelectbox > label {
            font-weight: bold;
        }
    </style>
""", unsafe_allow_html=True)

# ----------- SECTION 1: Brand & Display -----------
with st.expander("📦 Mobile Brand & Display Details", expanded=True):

    col1, col2 = st.columns(2)
    with col1:
        Brand = st.selectbox('Brand', df['Brand'].unique())
        color = st.selectbox('Color Choice', df['color'].unique())
    with col2:
        Model = st.selectbox('Mobile Category', df['Model'].unique())
        Display_Inch = st.number_input('Display size (in Inch)', min_value=3.0, max_value=7.5, step=0.1)
    
# ----------- SECTION 2: Memory & Processor -----------
with st.expander("⚙️ Memory & Processor Details", expanded=True):
    col1, col2 = st.columns(2)
    with col1:
        Processor = st.selectbox('Processor', df['Processor'].unique())
        Battery_Capacity = st.selectbox('Battery Size (mAh)', df['Battery_Capacity'])
    with col2:
        Mobile_RAM = st.selectbox('RAM (in GB)', [0, 2, 3, 4, 6, 8, 10, 12, 16, 32])
        ROM = st.selectbox('ROM (in GB)', [2, 3, 4, 8, 12, 16, 32, 64, 128, 256, 512, 1024])

# ----------- SECTION 3: Camera Configuration -----------
with st.expander("📷 Camera Configuration", expanded=True):
    col1, col2 = st.columns(2)
    with col1:
        Primary_rear_Camera = st.selectbox('Primary Rear Camera (MP)', [0, 1.3, 2, 5, 8, 12, 24, 32, 48, 50, 64, 108, 120, 130, 200])
        Secondary_rear_Camera = st.selectbox('Secondary Rear Camera (MP)', [0, 2, 5, 8, 10, 12,24,32, 48, 50])
    with col2:
        Number_of_rear_Cameras = st.selectbox('No. of Rear Cameras', [1, 2, 3, 4, 6])
        Front_Camera = st.selectbox('Front Camera (MP)', [0, 2, 3, 5, 8, 10, 12, 16, 20, 32, 42, 50])

# ----------- SECTION 4: Additional Features -----------
with st.expander("🔧 Additional Features", expanded=True):
    col1, col2, col3 = st.columns(3)
    with col1:
        Warranty_Available = st.radio("Warranty Available", ["Yes", "No"], index=0, horizontal=True)
    with col2:
        AI_lens = st.radio("AI Lens Enabled", ["Yes", "No"], index=0, horizontal=True)
    with col3:
        Front_Dual_Camera = st.radio("Front Dual Camera", ["Yes", "No"], index=1, horizontal=True)

# ----------- Prediction Button -----------
# ----------- Prediction Button -----------
st.markdown("---")
show_importance = False  # initialize flag

if st.button('🔍 Predict Price'):
    with st.spinner('Predicting...'):
        # Encode binary inputs
        Warranty_Available = 1 if Warranty_Available == 'Yes' else 0
        AI_lens = 1 if AI_lens == 'Yes' else 0
        Front_Dual_Camera = 1 if Front_Dual_Camera == 'Yes' else 0

        # Create input array
        query_dict = {
            'Battery_Capacity': [Battery_Capacity],
            'Processor': [Processor],
            'Brand': [Brand],
            'Model': [Model],
            'Mobile_RAM': [Mobile_RAM],
            'ROM': [ROM],
            'Display_Inch': [Display_Inch],
            'Primary_rear_Camera': [Primary_rear_Camera],
            'Secondary_rear_Camera': [Secondary_rear_Camera],
            'Number_of_rear_Cameras': [Number_of_rear_Cameras],
            'AI_lens': [AI_lens],
            'Front_Camera': [Front_Camera],
            'Front_Dual_Camera': [Front_Dual_Camera],
            'Warranty_Available': [Warranty_Available]
        }

        query_df = pd.DataFrame(query_dict)

        predicted_price = np.exp(pipe.predict(query_df)[0])

        # Display prediction result
        st.markdown(
            f"""
            <div style='text-align: center; font-size: 1.5rem; color: green; background-color: #e6f4ea; padding: 1rem; border-radius: 10px; border: 1px solid #a3d9a5; margin-top: 1rem;'>
                🎯 Predicted Price: ₹{predicted_price:,.2f}
            </div>
            """,
            unsafe_allow_html=True
        )

        show_importance = True  # set flag to show feature importances

if show_importance:
    # Load feature importance data
    combined_importance_df = joblib.load('combined_importance_df.joblib')

    with st.expander("📊 Feature Importances"):
        st.markdown("### 🔹 Feature Importances")
        st.bar_chart(combined_importance_df.set_index('Original_Feature'))



# Footer
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: gray;'>Made with ❤️ by Kedar</p>", unsafe_allow_html=True)

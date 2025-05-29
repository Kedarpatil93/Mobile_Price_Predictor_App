
#  Mobile Price Predictor

An interactive machine learning web app that predicts smartphone prices based on specifications like brand, RAM, ROM, display size, and camera configurations.


## Project Summary
- Built using XGBoost regression and feature engineering on smartphone data scraped from Flipkart (~1,000 entries).
- Achieved **R² score of 0.73** on test data through extensive tuning and validation.
- Integrated target encoding and a custom pipeline for clean input handling.
- Deployed as a user-friendly **Streamlit web app** for real-time predictions.


## Try the App
 [Launch Web App](https://mobile-price-predictor-kp123.streamlit.app/)


## How to Use
1. Open the [Mobile Price Predictor App](https://mobile-price-predictor-kp123.streamlit.app/)
2. Select mobile specifications (e.g., brand, RAM, display size, camera,..)
3. Click **Predict Price** to see thit e predicted value
4. View feature importance for model interpretability


## Features Used
- Brand, Model, RAM, ROM, Display Size
- Primary & Secondary Rear Camera, Front Camera
- AI lens, Dual Front Camera, Warranty, Battery Capacity


##  Potential Use Cases
-  **E-commerce platforms**: Price benchmarking for smartphone listings
-  **Market analysts**: Feature-to-price correlation analysis
-  **Sellers & buyers**: Evaluate price justification before transactions


##  What I Built
- Cleaned and preprocessed raw Flipkart smartphone listings
- Applied target encoding awith `category_encoders` and OneHot encoding and built a modular ML pipeline
- Trained and optimized different regression models (linear and tree-based) with cross-validation
- Finalised best performance model XGBoost regression based on evaluation metrics MAE and R2.  
- Designed and deployed a responsive UI using **Streamlit**


## Tech Stack
- Python, Pandas, NumPy, scikit-learn, XGBoost, joblib
- category_encoders, Streamlit

## Project Structure

<pre lang="markdown"> ```text mobile_price_predictor/ ├── main.py ├── pipe.joblib ├── df.joblib ├── combined_importance_df.joblib ├── data/ │ └── flipkart_mobiles.csv ├── notebook/ │ └── Mobile_price_prediction_trained_model.ipynb ├── requirements.txt ├── README.md └── .gitignore ``` </pre>

## Project Structure

```text
mobile_price_predictor/
├── main.py
├── pipe.joblib
├── df.joblib
├── combined_importance_df.joblib
├── data/
│   └── flipkart_mobiles.csv
├── notebook/
│   └── Mobile_price_prediction_trained_model.ipynb
├── requirements.txt
├── README.md
└── .gitignore

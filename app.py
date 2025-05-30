import streamlit as st
import pandas as pd
import numpy as np
import joblib
import re
from sklearn.preprocessing import MultiLabelBinarizer
import requests
from datetime import datetime

# Function to get current exchange rates
def get_exchange_rates():
    try:
        # Using ExchangeRate-API (you might want to use a proper API key in production)
        response = requests.get("https://api.exchangerate-api.com/v4/latest/TRY")
        if response.status_code == 200:
            data = response.json()
            print("Exchange rates fetched successfully.", data['rates'])
            return data['rates']
        else:
            st.warning("Couldn't fetch live exchange rates. Using default rates.")
            return {
                'USD': 0.033,
                'EUR': 0.030,
                'GBP': 0.026,
                'JPY': 4.92,
                'AED': 0.12
            }
    except:
        st.warning("Couldn't fetch live exchange rates. Using default rates.")
        return {
            'USD': 0.033,
            'EUR': 0.030,
            'GBP': 0.026,
            'JPY': 4.92,
            'AED': 0.12
        }
# Load saved artifacts
def load_artifacts():
    model = joblib.load('models/laptop_price_model.pkl')
    preprocessor = joblib.load('models/preprocessor.joblib')
    mlb = joblib.load('models/multilabel_binarizer.joblib')
    feature_columns = joblib.load('models/feature_columns.joblib')
    return model, preprocessor, mlb, feature_columns

# Preprocessing functions matching Colab notebook
def parse_storage(val):
    if isinstance(val, str):
        if 'TB' in val:
            return int(float(val.replace('TB', '').strip()) * 1024)
        elif 'GB' in val:
            return int(val.replace('GB', '').strip())
    return int(val)

def parse_processor_generation(val):
    return int(re.search(r'\d+', str(val)).group(0) if val else 0)

def preprocess_input(user_input, preprocessor, mlb, feature_columns):
    # Convert connections to list
    connections = user_input.pop('Connections')
    
    # Create DataFrame
    df = pd.DataFrame([user_input])
    
    # Apply same parsing as Colab notebook
    df['SSD Capacity'] = df['SSD Capacity'].apply(parse_storage)
    df['RAM (System Memory)'] = df['RAM (System Memory)'].apply(parse_storage)
    df['Graphics Card Capacity'] = df['Graphics Card Capacity'].apply(parse_storage)
    df['Processor Generation'] = df['Processor Generation'].apply(parse_processor_generation)
    df['Screen Size'] = df['Screen Size'].astype(float)
    
    # Process categorical columns
    categorical_columns = ['Brand', 'Processor Type', 'Operating System', 'Graphics Card',
                          'Graphics Card Memory Type', 'Graphics Card Type', 'Warranty Type',
                          'RAM (System Memory) Type', 'Processor Model', 'Usage Purpose']
    
    # Process numerical columns
    numerical_columns = ['SSD Capacity', 'RAM (System Memory)', 'Graphics Card Capacity',
                        'Processor Core Count', 'Processor Generation', 'GPU Memory',
                        'Screen Size', 'Maximum Expandable Memory']
    
    # Transform using the preprocessor
    processed_data = preprocessor.transform(df)
    
    # Get feature names
    numeric_feature_names = numerical_columns
    categorical_feature_names = preprocessor.named_transformers_['cat'].named_steps['onehot'].get_feature_names_out(categorical_columns)
    all_feature_names = list(numeric_feature_names) + list(categorical_feature_names)
    
    # Create DataFrame from processed data
    processed_df = pd.DataFrame(processed_data, columns=all_feature_names)
    
    # Process connections
    connections_encoded = pd.DataFrame(mlb.transform([connections]), columns=mlb.classes_)
    
    # Combine all features
    final_df = pd.concat([processed_df, connections_encoded], axis=1)
    
    # Ensure correct feature order
    final_df = final_df.reindex(columns=feature_columns, fill_value=0)
    
    return final_df

def get_user_input():
    with st.form("user_inputs"):
        # System Specifications
        st.header("System Specifications")
        col1, col2 = st.columns(2)
        
        with col1:
            brand = st.selectbox("Brand", options=sorted([
                'XASER', 'DMC', 'Zeiron', 'jetucuzal', 'Life Technology', 'IZOLY', 'TURBOX', 'Super', 'RAMTECH', 
                'Gamepage', 'Apple', 'GAMELINE', 'MSI', 'METSA', 'PCDEPO', 'Quantum Gaming', 'Canar', 'Gigabyte', 
                'ROGAME', 'LENOVO', 'EFS TECHNOLOGY', 'HP', 'OEM', 'ASUS', 'OXpower', 'ARTITEKNIKPC', 'Assembly', 
                'Güneysu Gaming', 'CASPER', 'UCARTECH', 'Technopc', 'DAGMOR', 'WARBOX', 'Avantron', 'Revenge', 
                'ColdPower', 'SECLIFE', 'TRINITY', 'Zetta', 'Corsair', 'RaXius', 'Oksid Information Technology', 
                'Tiwox', 'Jedi', 'Dell', 'Quadro', 'Rexdragon', 'Grundig', 'Redrock', 'Gaming Game', 'ACER', 'Tiranozor'
            ]), index=43)

            processor_type = st.selectbox("Processor Type", options=[
                'Intel Core i5', 'Intel Core i7', 'AMD', 'Intel Core i3', 'AMD Ryzen 5', 
                'AMD Ryzen 9', 'M2', 'Apple M1', 'Intel Pentium', 'AMD Ryzen 7', 'Intel Xeon', 
                'Intel Core i9', 'AMD Ryzen 3'
            ], index=0)

            ssd = st.selectbox("SSD Capacity", options=sorted([
                0, 8, 32, 120, 128, 240, 250, 256, 480, 500, 512, 1024, 2048, 4096
            ]), index=10)

            ram = st.selectbox("RAM (GB)", options=sorted([
                4, 8, 12, 16, 20, 24, 32, 36, 40, 48, 64, 96, 128, 192, 256
            ]), index=3)

        with col2:
            graphics_card = st.selectbox("Graphics Card", options=[
                'AMD Radeon RX 550', 'Nvidia Geforce GT 740', 'AMD Radeon RX 580', 'AMD Radeon RX550', 
                'Intel HD Graphics', 'Nvidia GeForce GTX 1650', 'Integrated Graphics', 
                'Nvidia GeForce RTX 3050', 'AMD Radeon RX 560', 'Nvidia GeForce GT 730', 
                'AMD Radeon R7 240', 'Intel Iris Graphics', 'AMD Radeon R5 230', 'Intel UHD Graphics 600', 
                'Nvidia GeForce GTX 1660 SUPER', 'Nvidia GeForce RTX 4070Ti SUPER', 'Nvidia GeForce RTX3060', 
                'Nvidia GeForce RTX 4060Ti', 'Nvidia GeForce GTX1050 Ti', 'Nvidia GeForce RTX 4070', 
                'Intel UHD Graphics 730', 'AMD Integrated Graphics', 'Nvidia Geforce GT 710', 
                'Nvidia GeForce RTX 4070Ti', 'Intel UHD Graphics 770', 'Nvidia GeForce RTX 4090Ti', 
                'AMD Radeon Graphics', 'AMD Radeon R5', 'Nvidia GeForce RTX 4060', 'AMD Radeon RX 550X', 
                'Intel UHD Graphics 630', 'Nvidia GeForce RTX 3080', 'Nvidia GeForce MX110', 
                'Nvidia GeForce RTX 3070', 'Intel HD Graphics 5500', 'NVIDIA GeForce GT1030', 
                'Nvidia Quadro T400', 'Nvidia RTX A2000', 'Nvidia RTX A4000', 'NVIDIA Quadro RTX 4000', 
                'AMD Radeon RX 570', 'Nvidia RTX A5000', 'Nvidia Quadro T600', 'Intel HD Graphics 4400', 
                'Nvidia Quadro T1000', 'Nvidia GeForce GTX 750', 'Nvidia GeForce GT 520M', 
                'Nvidia GeForce GTX1650 Ti', 'NVIDIA RTX A2000'
            ], index=3)

            graphics_capacity = st.selectbox("Graphics Card Capacity", options=sorted([
                8, 16, 64, 128, 250, 256, 320, 500, 512, 1024, 2048, 3072, 4096, 5120
            ]), index=8)

            os = st.selectbox("Operating System", options=[
                'Free Dos (No Operating System)', 'Windows', 'Mac Os', 'Ubuntu', 'Linux'
            ], index=1)

        # Advanced Specifications
        st.header("Advanced Specifications")
        col3, col4 = st.columns(2)
        
        with col3:
            graphics_memory_type = st.selectbox("Graphics Memory Type", options=[
                'GDDR5', 'DDR3', 'Integrated', 'GDDR6', 'DDR', 'DDR5', 'DDR4', 'GDDR6X', 'GDDR5X', 'SD', 'DDR2 + DDR3'
            ], index=0)

            graphics_type = st.selectbox("Graphics Card Type", options=['External', 'Integrated'], index=0)

            warranty = st.selectbox("Warranty Type", options=[
                'Official Distributor Warranty', 'Zeiron Turkey Warranty', 'Importer Warranty', 
                'Acer Turkey Warranty', 'Izoly Turkey Warranty', 'Apple Turkey Warranty', 
                'Technopc Turkey Warranty', 'Lenovo Turkey Warranty', 'Asus Turkey Warranty', 
                'HP Turkey Warranty', 'Casper Turkey Warranty', 'Samsung TR Warranty', 'Dell Turkey Warranty'
            ], index=3)

            ram_type = st.selectbox("RAM Type", options=['DDR3', 'DDR4'], index=1)

            processor_cores = st.selectbox("Processor Cores", options=sorted([
                0, 1, 2, 4, 6, 8, 10, 14, 16, 20, 24
            ]), index=3)

        with col4:
            processor_gen = st.selectbox("Processor Generation", options=sorted([
                1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14
            ]), index=11)

            processor_model = st.selectbox("Processor Model", options=[
                '3470', 'i5-3470', '4590', '9400F', '12400', '2400', '12400F', '860', '7400', 
                'Ryzen 5 5600', '3770', '2100', '3600', '4570', '300U', '2600', '10100F', '650', 
                '5900X', '10400F', '7600X', '5500U', '4440', '7800X', '7500U', '9400', '3700U', 
                '12700', '3220', 'i5-650', '7500', 'Xeon Silver 4214', '1000M', '10510U', '13700F', 
                '4770', '14700KF', '12100', '13500', '4800HS', '9100T', '6006U', '3240', '13400F', 
                '6400', '9900', '10100', '14700K', '3350P', '12500', '6100', '14900KF', '4405U', 
                '10400', '530', '7100', '11400', '3500U', '4150', '2200', '12700F', '13400', '6600H', 
                '8100', '13700', '12700K', '5300G', '4460', '5600H', '11700', '1235U', '8600K', 
                '8400', '10700', '560M', '8700', '12600K', '13100', '14900K', 'E52683', '8650U', '6500'
            ], index=4)

            usage = st.selectbox("Usage Purpose", options=[
                'Office - Work', 'Gaming', 'Home - School', 'Design'
            ], index=0)

            gpu_mem = st.selectbox("GPU Memory (GB)", options=sorted([0, 4, 6, 8, 12, 16, 24]), index=1)

            connections = st.multiselect("Connections", options=sorted([
                'USB', 'HDMI', 'Display Port', 'Kablo', 'VGA', 'Wi-Fi', 'oth', 'DVI'
            ]), default=['USB'])

        # Display and Additional Features
        st.header("Display & Memory")
        screen_size = st.selectbox("Screen Size (inches)", options=sorted([
            5.0, 5.5, 6.0, 18.0, 19.0, 20.0, 21.5, 22.0, 23.8, 24.0, 27.0
        ]), index=8)

        max_memory = st.selectbox("Max Expandable Memory (GB)", options=sorted([
            4, 8, 16, 32, 64, 128, 256
        ]), index=4)

        submitted = st.form_submit_button("Predict Price")
        
    if submitted:
        return {
            "Brand": brand,
            "Processor Type": processor_type,
            "SSD Capacity": ssd,
            "RAM (System Memory)": ram,
            "Graphics Card": graphics_card,
            "Graphics Card Capacity": graphics_capacity,
            "Operating System": os,
            "Graphics Card Memory Type": graphics_memory_type,
            "Graphics Card Type": graphics_type,
            "Warranty Type": warranty,
            "RAM (System Memory) Type": ram_type,
            "Processor Core Count": processor_cores,
            "Processor Generation": processor_gen,
            "Processor Model": processor_model,
            "Usage Purpose": usage,
            "GPU Memory": gpu_mem,
            "Screen Size": screen_size,
            "Maximum Expandable Memory": max_memory,
            "Connections": connections
        }
    return None

# Main app
def main():
    st.title("Computer Price Predictor")
    
    # Load artifacts
    model, preprocessor, mlb, feature_columns = load_artifacts()
    
    # Get user input
    user_input = get_user_input()
    
    if user_input:
        # Preprocess input
        processed_data = preprocess_input(user_input, preprocessor, mlb, feature_columns)
        
        # Make prediction
        prediction = model.predict(processed_data)[0]

        # Convert prediction to Turkish Lira (TRY) if needed
        exchange_rates = get_exchange_rates()
        
        # Display results
        st.subheader("Prediction Result")
        col1, col2 = st.columns([1,2])
        with col1:
            st.metric(label="Predicted Price (TRY)", value=f"{prediction:,.2f} ₺")
        with col2:
            st.markdown("Converted to other currrency rates:")
            currencies = {
                'INR': '₹',
                'USD': '$',
                'EUR': '€',
                'AED': 'AED',
                'JPY': '¥',
                'GBP': '£',
            }
            for currency, symbol in currencies.items():
                if currency in exchange_rates:
                    converted_price = prediction * exchange_rates[currency]
                    st.metric(label=f"Predicted Price ({symbol})", value=f"{converted_price:,.2f} {symbol}")
        st.caption(f"Exchange rates are fetched live as of {datetime.now().strftime('%d-%m-%Y %H:%M')}.")
        
        # Show raw input for verification
        st.subheader("Processed Input Features")
        st.write(processed_data)

if __name__ == "__main__":
    main()
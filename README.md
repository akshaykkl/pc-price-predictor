````markdown
# 💻 PC Price Predictor

A **Streamlit web app** that predicts **PC prices** based on hardware specifications like CPU, GPU, RAM, etc., using a machine learning model trained on a Turkish dataset.

## 🚀 Features

- 🔍 Predict PC Prices based on key hardware specs
- 🧠 Machine Learning Powered predictions
- 📊 Trained on real-world Turkish PC listings
- 🖱️ Interactive & intuitive UI with Streamlit
- 🧩 Easily expandable with new features (e.g., SSD/HDD, brand, screen size)
- 💱 Available currency transformations to some of the common countries.

## 📦 Installation

1. Clone the repository

```bash
git clone https://github.com/akshaykkl/pc-price-predictor.git
cd pc-price-predictor
```

2. Install the dependencies

```bash
pip install -r requirements.txt
```

3. Run the Streamlit app

```bash
streamlit run app.py
```

## 🧑‍💻 Usage

After launching the app, fill in the hardware specifications:

- CPU model
- GPU model
- RAM size
- Storage type and size
- Screen resolution 

Click **Predict Price** to get an estimated PC price in Turkish Lira (₺).

Optionally, explore how changing different components affects the price.

## 🧠 Model Details

- **Algorithm:** [e.g., Random Forest Regressor / XGBoost / Linear Regression]
- **Dataset Source:** Turkish e-commerce PC listings
- **Preprocessing:** Cleaned data, label encoding for categorical features, feature scaling

### Performance:

- **R² Score:** [0.919807398421276]
- **MAE:** [11798427.2306756]

### Features Used:

- CPU
- GPU
- RAM size
- Storage (type & size)
- Screen resolution



## ⚠️ Notes

- The model is trained on Turkish-market data, so predictions are tuned to prices in Turkey and may not generalize globally.
- Ensure all component names are entered correctly for accurate predictions.

## 💡 Future Improvements

- Add currency converter for international users
- Include more brands and newer models
- Add charts and analytics for price trends
- Deploy the app online with Streamlit Cloud or Hugging Face Spaces

## 🤝 Contributing

Contributions are welcome!

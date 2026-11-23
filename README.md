# Credit Card Fraud Detection System

An AI-powered application built to detect and prevent fraudulent credit card transactions using Deep Learning and Machine Learning techniques.

## Features
- **Real-Time Fraud Prediction:** Evaluates transaction details to classify activity as legitimate or fraudulent.
- **Pre-Trained Deep Learning Model:** Uses a saved Keras neural network (`fraud_model.keras`) for fast inference.
- **Consistent Data Preprocessing:** Applies a fitted Scikit-learn scaler (`scaler.pkl`) to normalize transaction inputs accurately.
- **Interactive Web Interface:** Built with Python (`app.py`) for easy testing and visualization.

## Project Structure
```text
Credit-Card-Fraud-Detection-/
├── app.py               # Main application script
├── fraud_model.keras    # Trained Deep Learning model
└── scaler.pkl           # Feature scaler for preprocessing input data
```

## Tech Stack
- **Language:** Python
- **Deep Learning:** TensorFlow / Keras
- **Data Processing:** Scikit-learn, NumPy, Pandas
- **Web App Framework:** Streamlit

## How to Run Locally

1. **Clone the repository:**
   ```bash
   git clone https://github.com
   cd Credit-Card-Fraud-Detection-
   ```

2. **Create and activate a virtual environment:**
   ```bash
   python -m venv .venv
   # On Windows:
   .venv\Scripts\activate
   # On Mac/Linux:
   source .venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install streamlit tensorflow scikit-learn pandas numpy
   ```

4. **Run the application:**
   ```bash
   streamlit run app.py
   ```

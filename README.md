# 🏭 Machine Downtime Prediction Project  

A predictive maintenance project that leverages **ARIMA time-series modeling** to forecast machine downtimes based on historical loss data.  

This application, built with **Streamlit**, allows users to interactively select a machine downtime reason code and predict the next downtime occurrence in hours.  

---

## 🚀 Features
- ✅ Forecasts machine downtimes using **ARIMA models**  
- ✅ Interactive **Streamlit UI** for easy exploration  
- ✅ Handles multiple **Loss Reason Codes** dynamically  
- ✅ Resamples downtime logs into **hourly intervals**  
- ✅ Provides predictions for the **next 7 days (168 hours)**  

---


## 🛠️ Tech Stack
- **Python 3.10+**  
- **pandas, numpy** (data preprocessing)  
- **pmdarima (auto_arima)** for ARIMA modeling  
- **scikit-learn** (evaluation metrics)  
- **Streamlit** (web app interface)  

---

## ⚙️ Setup Instructions

### 1. Clone the repository

git clone https://github.com/your-username/machine_downtime_prediction_project.git
cd machine_downtime_prediction_project

### 2. Create a virtual environment

python -m venv venv
source venv/bin/activate   # Linux/Mac
venv\Scripts\activate      # Windows

### 3. Install dependencies

pip install -r requirements.txt

### 4. Add your dataset

Place your downtime dataset as downtime_data.csv in the project root folder.

### 5. Run the app
streamlit run app.py



## 👤 Author
**Siddharth Bhimpure**  
- 🎓 B.Tech in AI & Data Science  

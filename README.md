# 💳 Credit Card Fraud Detection using Machine Learning & Data Processing Pipeline

This project combines **Data Processing** and **Machine Learning** to detect fraudulent credit card transactions from a highly imbalanced dataset containing **284,807 financial records**.

The workflow includes **data cleaning, data transformation, feature engineering, validation, SMOTE-based class balancing, and predictive modeling** using multiple machine learning algorithms to accurately identify fraudulent transactions.

The project demonstrates an end-to-end data preprocessing workflow before applying machine learning models for fraud detection.


### 🚀 Project Objective
To build a reliable data preprocessing and fraud detection pipeline that cleans, transforms, validates, and analyzes financial transaction data before training machine learning models for accurate fraud detection.


### 📂 Dataset

- **Source:** Kaggle – Credit Card Fraud Detection Dataset  
- **Total Transactions:** 284,807  
- **Fraud Cases:** Only **492 (0.17%)**  
- **Link:** https://www.kaggle.com/mlg-ulb/creditcardfraud  
- Note: Features are PCA-transformed except **`Time`** and **`Amount`**.



### 🛠️ Technologies & Techniques Used

| Category | Tools |
|----------|-------|
| **Programming** | Python |
| **Libraries** | Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn |
| **Data Processing** | Data Cleaning, Data Transformation, Feature Engineering, StandardScaler |
| **Machine Learning** | Logistic Regression, Random Forest, XGBoost, SMOTE |
| **Evaluation** | Confusion Matrix, ROC-AUC, Classification Report |



## 📊 Project Workflow
Credit Card Dataset
        ↓
Data Cleaning
        ↓
Data Transformation
        ↓
Feature Engineering
        ↓
Data Validation
        ↓
StandardScaler
        ↓
SMOTE
        ↓
Machine Learning Models
        ↓
Performance Evaluation


### 📈 Modeling & Evaluation

✔ Used **20% stratified sample** for faster training  
✔ Handled **severe class imbalance (0.17%)** using SMOTE  
✔ Trained 3 models & compared performance

| Model | AUC Score | Fraud Class F1-Score |
|-------|----------|----------------------|
| ⭐ Logistic Regression | **0.978 (Best)** | 0.25 |
| Random Forest | 0.900 | 0.81 |
| XGBoost | 0.950 | 0.81 |

🏆 **Best Model Selected:** Logistic Regression  
📌 *Selected due to superior AUC & reliable decision boundary on imbalanced data.*


### 📊 Output Visualizations

All generated plots are automatically saved in the **`output/`** directory:

🔹 Class Imbalance  
🔹 Correlation Heatmap  
🔹 Amount vs. Fraud Box Plot  
🔹 Time Distribution Analysis  
🔹 Confusion Matrix (each model)  
🔹 ROC–AUC Curve Comparison  
🔹 Feature Importance (if Random Forest wins)

> 🎨 Graphs are saved in High-Resolution `.png` format

### ▶️ How to Run the Project

#### 1️⃣ Install Required Libraries
pip install -r requirements.txt

2️⃣ Run the Script
python main.py

3️⃣ Check Results

Model training summary shown in terminal

Predictions printed for single transaction

Generated plots are stored in output/

📦 Project Structure
CreditCard-Fraud-Detection/
│
├── creditcard.csv / dataset/ (dataset)
├── main.py
├── requirements.txt
├── output/ (generated automatically)
└── README.md

🔮 Future Enhancements

🚀 Deploy using Flask / FastAPI
🔐 Real-time fraud scoring API
📊 Dashboard for transaction monitoring
💾 Store flagged fraud records in MySQL/PostgreSQL for reporting and analytics


## 📌 Key Skills Demonstrated
Python • Data Cleaning • Data Processing • Data Transformation • Feature Engineering • Machine Learning • Data Analysis • Model Evaluation • Data Visualization


👨‍💻 Author

Neeraj Chauhan
💼 Data Science & Machine Learning Enthusiast
🌐 GitHub: https://github.com/neerajchauhan98

⭐ If you like this project, don’t forget to star the repository!

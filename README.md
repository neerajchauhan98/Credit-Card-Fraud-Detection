# 💳 Credit Card Fraud Detection using Machine Learning

A machine learning–based fraud detection system designed to identify suspicious credit card transactions in highly imbalanced datasets.  
This project applies statistical analysis, data preprocessing, oversampling techniques, and classification models to detect fraudulent activities in financial transactions.

---

### 🚀 Project Objective
To develop an intelligent fraud detection model that identifies high-risk credit card transactions using advanced machine learning algorithms, ensuring secure and trustworthy digital payments.

---

### 📂 Dataset

- **Source:** Kaggle – Credit Card Fraud Detection Dataset  
- **Total Transactions:** 284,807  
- **Fraud Cases:** Only **492 (0.17%)**  
- **Link:** https://www.kaggle.com/mlg-ulb/creditcardfraud  
- Note: Features are PCA-transformed except **`Time`** and **`Amount`**.

---

### 🛠️ Technologies & Techniques Used

| Category | Tools |
|----------|-------|
| **Languages** | Python |
| **Libraries** | Pandas, NumPy, Matplotlib, Seaborn |
| **ML Algorithms** | Logistic Regression, Random Forest, XGBoost |
| **Preprocessing** | StandardScaler, SMOTE (Oversampling) |
| **Evaluation Metrics** | Confusion Matrix, ROC-AUC, Classification Report |

---

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

---

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

---

### ▶️ How to Run the Project

#### 1️⃣ Install Required Libraries
```bash
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
💾 Store flagged fraud records in database systems

👨‍💻 Author

Neeraj Chauhan
💼 Data Science & Machine Learning Enthusiast
🌐 GitHub: https://github.com/neerajchauhan98

⭐ If you like this project, don’t forget to star the repository!
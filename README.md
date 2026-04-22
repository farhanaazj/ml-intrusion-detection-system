# 🔐 ML-Based Intrusion Detection System for Network Traffic Anomaly Detection

This project evaluates the effectiveness of machine learning-based anomaly detection techniques for identifying malicious network traffic. Multiple classification models are implemented and compared to improve intrusion detection performance, with a focus on achieving high accuracy and minimising false-positive rates in real-world applications.

## 🛠️ Tech Stack
- Python
- Scikit-learn
- Pandas, NumPy
- Matplotlib, Seaborn
- Streamlit

## 📌 Project Objectives
- Review existing intrusion detection systems and identify research gaps
- Preprocess and analyse network traffic data for anomaly detection
- Implement and evaluate multiple machine learning models for intrusion detection
- Assess the impact of feature selection and dimensionality reduction on model performance
- Evaluate models using accuracy, F1-score, ROC-AUC, and false-positive rate

## 📂 Dataset Description
Source: Mendeley Data
https://data.mendeley.com/datasets/4pnwdgt7b7/1

Dataset Details:
- Rows: 10,005  
- Columns: 12  
- Type: Binary classification (Normal vs Attack traffic)  
- Target Variable: Traffic label (0 = Normal, 1 = Attack)  
  - Key Features: Source and destination ports, Network protocol and connection duration, Packet count and traffic volume (bytes sent/received), Bytes per packet and flow-level characteristics, Encoded IP addresses and timestamp information

## 🤖 Machine Learning Models Used

### 1️⃣ Logistic Regression
- Baseline linear classification model  
- Simple and interpretable  
- Limited in capturing complex patterns  

### 2️⃣ Random Forest
- Ensemble learning model using multiple decision trees  
- Handles non-linear relationships effectively  
- Best-performing model in this project  

### 3️⃣ Gradient Boosting
- Boosting-based ensemble model  
- Improves predictions by focusing on misclassified samples  
- Strong performance, close to Random Forest

## 🧪 Model Evaluation
- Accuracy, Precision, Recall, F1-Score, ROC-AUC Score, False Positive Rate, Confusion Matrix

## 📈 Results & Insights
- Random Forest achieved the highest accuracy and lowest false-positive rate
- Gradient Boosting delivered similarly strong results
- Ensemble models significantly outperformed Logistic Regression
- Feature engineering significantly improved detection accuracy and reduced false positives

## 📊 Visualizations
- Class distribution of network traffic
- Correlation matrix of top features
- Feature distribution by class (boxplots)
- Model performance comparison charts
- ROC curves for all models
- Confusion matrices for all models
- Feature importance analysis
- Hyperparameter tuning results
- Interactive Streamlit dashboard for live predictions on normal and attack traffic

## ▶️ How to Run
- Clone the repository
git clone https://github.com/farhanaazj/ml-intrusion-detection-system
- Install dependencies
pip install -r requirements.txt
- Run the Streamlit app
streamlit run app.py
- Upload **network_traffic.csv** in the sidebar
- Click **▶ Run Analysis** to train models and explore results
- The app will automatically open in your browser
   - If not, go to: http://localhost:8501

## 🔮 Future Work
- Evaluate models across multiple and diverse datasets to improve generalisation
- Incorporate deep learning models (CNN, LSTM) for detecting complex and temporal attack patterns
- Explore unsupervised and semi-supervised methods for identifying unknown attack types
- Develop real-time IDS with streaming data and SIEM system integration
- Apply explainable AI techniques to improve model interpretability and trust

## 👩‍💻 Author
Farhanaaz J

Data Science Project - Dissertation

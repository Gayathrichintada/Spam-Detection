# 📧 Spam Message Detection using Machine Learning

## 📌 Project Overview
This project implements a **Spam Message Detection System** using **Machine Learning and Natural Language Processing (NLP)** techniques.  
The system classifies an input SMS or email message into one of the following categories:

- ✅ Not Spam  
- ⚠️ Suspicious  
- 🚨 Spam  

The project also includes a **web-based interface** built using **Streamlit**, allowing users to test messages in real time.

---

## 🎯 Objectives
- To detect spam messages using machine learning techniques
- To preprocess and analyze text data using NLP
- To classify messages with confidence scores
- To build a user-friendly web application for real-time prediction

---

## 🧠 Machine Learning Approach

### 🔹 Type of Learning
- **Supervised Learning**
- **Text Classification**

### 🔹 Algorithm Used
- **Multinomial Naive Bayes**

### 🔹 Feature Extraction
- **TF-IDF (Term Frequency–Inverse Document Frequency)**
- **N-grams (unigram + bigram)** for improved detection of subtle spam

---

## 🛠️ Technologies Used
- Python
- Pandas, NumPy
- Scikit-learn
- NLTK
- Streamlit
- Joblib (for model saving)

---

## 📂 Dataset
- **SMS Spam Collection Dataset**
- Source: UCI / Kaggle
- Contains labeled messages:
  - `ham` → Not Spam
  - `spam` → Spam

---

## ⚙️ Project Workflow
1. Load and clean the dataset
2. Text preprocessing:
   - Lowercasing
   - Punctuation removal
   - Stopword removal
   - Stemming
3. Feature extraction using TF-IDF with n-grams
4. Model training using Naive Bayes
5. Saving trained model and vectorizer
6. Building a Streamlit web application
7. Real-time prediction with confidence score

---

## 🌐 Web Application Features
- Input any SMS or email text
- Real-time classification
- Confidence score display
- Three-level output:
  - Not Spam
  - Suspicious
  - Spam
- Fast performance using cached model loading

---

## 📁 Project Structure
Spam-Detection-ML/
│
├── app.py # Streamlit web application
├── train_and_save.py # One-time model training script
├── spam.csv # Dataset
├── spam_model.joblib # Saved ML model
├── tfidf.joblib # Saved TF-IDF vectorizer
├── README.md # Project documentation

yaml
Copy code

---

## ▶️ How to Run the Project

### Step 1: Install dependencies
```bash
pip install pandas numpy scikit-learn nltk streamlit joblib
Step 2: Train the model (run once)
bash
Copy code
python train_and_save.py
Step 3: Run the web application
bash
Copy code
streamlit run app.py
Open browser:

arduino
Copy code
http://localhost:8501
📊 Sample Output
Input:
"Your service eligibility has been updated. Please verify your information to avoid disruption."

Output:
⚠️ Suspicious (Confidence: 44%)

🚀 Enhancements Over Basic Spam Detection
Use of n-grams for improved detection

Hybrid ML + rule-based approach

Confidence-based classification

Suspicious message category

Web-based UI for real-time testing

Performance optimization using saved models

⚠️ Limitations
Subtle phishing messages without explicit spam keywords may still be difficult to detect

Model is trained mainly on SMS data, not full email datasets

🔮 Future Improvements
Use deep learning models (LSTM / BERT)

Train on phishing email datasets

Deploy application online

Multi-language spam detection

🎓 Academic Relevance
Suitable for 3rd Year B.Tech Mini Project

Demonstrates practical use of Machine Learning and NLP

Industry-relevant problem with real-world application

👨‍💻 Author
Chintada Gayathri

B.Tech (3rd Year)

Department of Computer Science and Engineering


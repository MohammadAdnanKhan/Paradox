# Paradox 🔮  
A machine learning-powered web app to classify YouTube comments into three categories: **Feedback**, **Doubt**, and **Irrelevant**. Paradox is designed to help content creators and educators organize and analyze comments efficiently.  

## 📑 Table of Contents  
1. [Paradox Overview](#paradox-)
2. [Features](#-features)
3. [How It Works](#-how-it-works)
4. [Dataset](#-dataset)
5. [Installation](#-installation)
6. [Link to Paradox](#-link-to-paradox)
7. [Model Details](#-model-details)
8. [Future Scope](#-future-scope)
9. [Project Motivation](#-project-motivation)
10. [Author](#-author)

## 🌟 Features  
- **Single Comment Prediction**: Classify individual YouTube comments in real-time.  
- **Batch Prediction**: Upload a file to classify multiple comments at once.  
- **Interactive Deployment**: The app is deployed using Streamlit for a seamless user experience.  
- **User-Friendly Interface**: Simple navigation and instant results.  

## 🚀 How It Works  
1. **Preprocessing**: Comments are cleaned, normalized, and stemmed using the Porter Stemmer.  
2. **Feature Extraction**: The text is transformed using a **TF-IDF Vectorizer** to convert it into a numerical format.  
3. **Model Prediction**: A **Logistic Regression** model predicts the category of the comments.  
4. **Label Encoding**: Predicted categories are decoded for easy interpretation.  

## 📊 Dataset  
Paradox was trained on a dataset of over **210,000 YouTube comments**, categorized into:  
- **Feedback**: Constructive comments providing feedback.  
- **Doubt**: Questions or clarifications sought by users.  
- **Irrelevant**: Comments unrelated to the content.  

## ⚙️ Installation
To run Paradox locally, follow these steps:

1. **📂 Clone the Repository**:
    ```bash
    git clone https://github.com/MohammadAdnanKhan/Paradox.git
    cd Paradox
    ```

2. **📦 Install Requirements**:
    ```bash
    pip install -r requirements.txt
    ```

3. **⬇️ Download the Model**:
   Place your trained model file (`Trained_model.pkl`), tfidf Transformer (`tfidf_transformer.pkl`), label encoder (`label_encoder.pkl`) in the project directory.

4. **🚀 Run the Application**:
    ```bash
    streamlit run script.py
    ```


## 🌐 Link to Paradox
Check out Paradox on Streamlit: [Paradox](https://paradoxxx.streamlit.app/).


## 📈 Model Details  
- **Algorithm**: Logistic Regression  
- **Preprocessing**: TF-IDF Vectorization and Porter Stemming  
- **Evaluation Metric**: Accuracy  

## 💡 Future Scope  
- Add more categories for nuanced classification.  
- Improve model accuracy with advanced techniques like transformer-based models.  
- Incorporate sentiment analysis for additional insights.  
- Enable multilingual comment classification.  

## 🔮 Project Motivation  
Paradox was created to help content creators organize and analyze YouTube comments more effectively, making it easier to manage audience feedback and insights.  

## 👤 Author
**Mohd Adnan Khan**  

- 💼 [LinkedIn](https://www.linkedin.com/in/mohd-adnan--khan)
- 🐙 [GitHub](https://github.com/MohammadAdnanKhan)
- 📊 [Kaggle](https://www.kaggle.com/mohdadnankhan1)
- **📧 Contact**: mohdadnankhan.india@gmail.com
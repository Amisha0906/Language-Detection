# 🌍 Language Detection Model  

This project is a Machine Learning model that detects the language of a given text. It is trained on a diverse dataset and supports multiple languages.  

## Model Download
The trained machine learning model (pickle file) is too large to be uploaded to GitHub. You can download it from Google Drive using the link below:

📥 [Download Model (language_detector_model.pkl)](https://drive.google.com/file/d/1Ne3HmqLuFGvHU1xgz4qRFul4HcBpQhW6/view?usp=drive_link)

## 📊 Dataset
The model is trained on a dataset containing text samples from 62 languages.
📥[ Dataset Download (language_dataset.csv)](https://drive.google.com/file/d/1BlvYVY3f4S8zU34njCJWlVmuhJX8Yq1F/view?usp=drive_link)

## Approach

### 🔧 Preprocessing

1. Converted all text to lowercase
2. Removed punctuation and digits
3. Striped extra spaces

### 🧠 Model Building
The model pipeline includes:

* TfidfVectorizer: For feature extraction (word/char n-grams)
* MultinomialNB: For classification
* Grid search was used to tune MultinomialNB model

## 🏆 Model Performance
Accuracy: 96.15%
Precision: 96.28%
Recall: 96.15%

## 🤝 Contributing
Contributions are welcome! Feel free to submit a pull request if you'd like to improve the model or add more features.

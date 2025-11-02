Data Science Intermediate Project
# Fake News Detection

## Project Overview
This project builds a Fake News Detection System trained on a dataset containing real and fake news articles. It uses TF-IDF vectorization and a Logistic Regression or Naive Bayes classifier to analyze the textual patterns of news content. The system is deployed with Streamlit for easy interaction — users can enter a news statement and instantly see if it’s fake or real, along with a confidence score.

## Key Features
- Text preprocessing with cleaning, stopword removal, and lemmatization
- TF-IDF vectorization for feature extraction
- Model trained to achieve around 98–99% accuracy
- Streamlit web interface for real-time predictions
- Confidence score displayed for each prediction
- Automatically loads a trained model (or trains one if not found)

## Why Some True Statements Are Flagged as Fake
Short or overly simple sentences such as "Donald Trump is a president of America" may be predicted as fake because the model was trained mostly on full-length news articles. It assumes "fake" when:
- The text resembles clickbait or incomplete statements seen in fake samples
- The statement lacks journalistic context such as sources or structure
- The model detects political figure mentions that often correlated with fake news in the dataset

To improve predictions, provide more context. For example:
"Donald Trump served as the 45th President of the United States, according to official records."
This provides linguistic structure and facts, which help the model classify more accurately.

## Tech Stack
- Python 3
- Pandas, NumPy
- NLTK (for stopwords and lemmatization)
- Scikit-learn (for TF-IDF and model training)
- Streamlit (for deployment)
- Joblib (for saving/loading models)

## Dataset 
link : https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset

## Folder Structure
Fake-News-Detection/
│
├── fake_news_detection.ipynb   # Model training and evaluation
├── app.py                      # Streamlit application
├── vectorizer.pkl              # Saved TF-IDF vectorizer
├── model.pkl                   # Saved ML model
├── true.csv                    # True news dataset
├── fake.csv                    # Fake news dataset
└── README.md                   # Project documentation

## How to Run
1. Install dependencies
   pip install -r requirements.txt

2. Run the Streamlit app
   streamlit run app.py

3. Interact with the app  
   Enter any news content and click "Check News" to get:
   - Real or Fake label
   - Confidence percentage

## Model Performance
Accuracy: 98.8%  
Precision: 0.99  
Recall: 0.99  
F1-score: 0.99  

## Example Predictions
Input: "The U.S. Senate passed a new infrastructure bill on Tuesday."  
Output: Real News (Confidence: 97%)

Input: "NASA confirms Earth will go dark for 15 days next month."  
Output: Fake News (Confidence: 99%)

## Future Improvements
- Add multiple ML models for comparison
- Include article source verification
- Integrate live fact-checking API

## Contributing
Contributions are welcome!
Feel free to fork the repository, improve the game, and open a pull request. Let's grow this classic game together!

## License
This project is licensed under the [![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](./LICENSE)

## Author
**Aarya Mehta**  
🔗 [GitHub Profile](https://github.com/AaryaMehta2506)



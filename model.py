import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
import joblib
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

def preprocess_text(text):
    text = text.lower()
    text = ''.join([char for char in text if char.isalpha() or char.isspace()])
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in text.split() if word not in stop_words]
    stemmer = PorterStemmer()
    tokens = [stemmer.stem(word) for word in tokens]
    return ' '.join(tokens)

def train_model():
    df = pd.read_excel("BankReviews.xlsx")
    df['processed_review'] = df['Reviews'].apply(preprocess_text)
    
    X_train, X_test, y_train, y_test = train_test_split(df['processed_review'], df['Stars'], test_size=0.2, random_state=42)

    model = make_pipeline(CountVectorizer(), MultinomialNB())
    model.fit(X_train, y_train)

    joblib.dump(model, 'sentiment_model.joblib')

if __name__ == "__main__":
    train_model()

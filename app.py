from flask import Flask, request, render_template
import joblib
from model import preprocess_text

app = Flask(__name__)

model = joblib.load('sentiment_model.joblib')

sentiment_mapping = {
    1: 'Negative review',
    2: 'Negative review',
    3: 'Neutral review',
    4: 'Positive review',
    5: 'Positive review'
}

def is_valid_review(text):
    # Check if the text is empty
    if not text:
        return False
    
    # List of valid keywords or phrases
    valid_keywords = ["great", "bad", "superb", "excellent", "awful", "poor", "wonderful", "horrible", "fantastic", "amazing", "terrible", "good", "satisfactory"]  # Add more as needed
    

    if any(keyword in text.lower() for keyword in valid_keywords):
        return True
    
    # Check if the text is too short
    if len(text.split()) < 3:
        return False
    
    return True


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        user_input = request.form['review']
        
        # Check if the review is valid
        if not is_valid_review(user_input):
            return render_template('index.html', error_message="Please enter a valid  review.", user_input=user_input)

        processed_input = preprocess_text(user_input)
        
        star_prediction = model.predict([processed_input])[0]
        sentiment_prediction = sentiment_mapping.get(star_prediction, 'Unknown')

        return render_template('index.html', star_prediction=star_prediction, sentiment_prediction=sentiment_prediction, user_input=user_input)

if __name__ == '__main__':
    app.run(debug=True)

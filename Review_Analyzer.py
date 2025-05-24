import pandas as pd
import numpy as np
import re
import string
from pathlib import Path

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.externals import joblib


class ReviewPreprocessor:
    """Handles text preprocessing for review analysis"""
    
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
        self.negation_words = {'not', 'no', "n't"}
    
    def preprocess_text(self, text):
        """Clean and normalize text for analysis"""
        if not isinstance(text, str):
            return ""
            
        # Tokenize with negation handling
        tokens = self._handle_negations(text.split())
        
        # Normalize tokens
        tokens = [
            self.lemmatizer.lemmatize(word.lower())
            for word in tokens
            if (word.lower() not in self.stop_words and 
                len(word) >= 3 and
                word not in string.punctuation)
        ]
        
        return ' '.join(tokens)
    
    def _handle_negations(self, tokens):
        """Handle negation words in token stream"""
        processed = []
        i = 0
        while i < len(tokens):
            if tokens[i] in self.negation_words and i < len(tokens) - 1:
                processed.append(f"not_{tokens[i+1]}")
                i += 2
            else:
                processed.append(tokens[i])
                i += 1
        return processed


class ReviewClassifier:
    """Trains and manages review sentiment classifier"""
    
    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=2900)
        self.model = MultinomialNB()
        self.preprocessor = ReviewPreprocessor()
    
    def train(self, X, y, test_size=0.25, random_state=0):
        """Train the classifier with given data"""
        print("Processing text data...")
        X_processed = [self.preprocessor.preprocess_text(text) for text in X]
        
        print("Splitting dataset...")
        X_train, X_test, y_train, y_test = train_test_split(
            X_processed, y, test_size=test_size, random_state=random_state
        )
        
        print("Vectorizing text...")
        train_X = self.vectorizer.fit_transform(X_train)
        test_X = self.vectorizer.transform(X_test)
        
        print("Training classifier...")
        self.model.fit(train_X, y_train)
        
        pred = self.model.predict(test_X)
        accuracy = accuracy_score(y_test, pred)
        print(f"Classifier accuracy: {accuracy:.2f}")
        
        return accuracy
    
    def predict(self, texts):
        """Predict sentiment for given texts"""
        processed = [self.preprocessor.preprocess_text(text) for text in texts]
        vectorized = self.vectorizer.transform(processed)
        return self.model.predict(vectorized)
    
    def save_model(self, filepath):
        """Save trained model to file"""
        joblib.dump({
            'model': self.model,
            'vectorizer': self.vectorizer
        }, filepath)
        print(f"Model saved to {filepath}")
    
    @classmethod
    def load_model(cls, filepath):
        """Load trained model from file"""
        data = joblib.load(filepath)
        classifier = cls()
        classifier.model = data['model']
        classifier.vectorizer = data['vectorizer']
        return classifier


class ReviewAnalyzer:
    """Analyzes product reviews and calculates ratings"""
    
    def __init__(self, classifier=None):
        self.classifier = classifier or ReviewClassifier()
        self.preprocessor = ReviewPreprocessor()
    
    def analyze_reviews(self, reviews_df, aspect_filters=None):
        """
        Analyze reviews and calculate overall and aspect-specific ratings
        
        Args:
            reviews_df: DataFrame containing reviews
            aspect_filters: Dict of {aspect_name: [keywords]} for aspect analysis
            
        Returns:
            Dict containing overall and aspect ratings
        """
        if 'Review Content' not in reviews_df.columns:
            raise ValueError("DataFrame must contain 'Review Content' column")
        
        # Overall analysis
        reviews = reviews_df['Review Content'].tolist()
        predictions = self.classifier.predict(reviews)
        overall_rating = self._calculate_rating(predictions)
        
        results = {
            'overall': {
                'rating': overall_rating,
                'prediction_count': len(predictions)
            }
        }
        
        # Aspect-specific analysis
        if aspect_filters:
            for aspect, keywords in aspect_filters.items():
                aspect_reviews = self._filter_aspect_reviews(reviews, keywords)
                if aspect_reviews:
                    aspect_pred = self.classifier.predict(aspect_reviews)
                    aspect_rating = self._calculate_rating(aspect_pred)
                    results[aspect] = {
                        'rating': aspect_rating,
                        'prediction_count': len(aspect_pred)
                    }
        
        return results
    
    def _filter_aspect_reviews(self, reviews, keywords):
        """Filter reviews mentioning specific aspect keywords"""
        aspect_reviews = []
        for review in reviews:
            sentences = re.split(r'[.!?]', review)
            for sentence in sentences:
                if any(keyword.lower() in sentence.lower() for keyword in keywords):
                    aspect_reviews.append(sentence)
        return aspect_reviews
    
    def _calculate_rating(self, predictions):
        """Convert prediction array to 5-star rating"""
        positive_count = sum(predictions)
        if not predictions:
            return 0
        return (positive_count / len(predictions)) * 5


def load_data(filepath, rating_threshold=3):
    """Load and prepare dataset from CSV"""
    data = pd.read_csv(filepath)
    
    # Convert ratings to binary (1=positive, 0=negative)
    data['Sentiment'] = (data['Rating'] >= rating_threshold).astype(int)
    
    return data['Reviews'], data['Sentiment']


def main():
    # Configuration
    TRAIN_DATA_PATH = Path("Amazon_Unlocked_Mobile.csv")
    REVIEWS_DATA_PATH = Path("Review(Content - Redmi Note 5 Pro).csv")
    MODEL_PATH = Path("review_classifier.pkl")
    ASPECT_FILTERS = {
        'camera': ['camera', 'photo', 'picture', 'image', 'selfie']
    }
    
    try:
        # Load and prepare training data
        print("Loading training data...")
        X, y = load_data(TRAIN_DATA_PATH)
        
        # Train or load classifier
        if MODEL_PATH.exists():
            print("Loading existing model...")
            classifier = ReviewClassifier.load_model(MODEL_PATH)
        else:
            print("Training new model...")
            classifier = ReviewClassifier()
            classifier.train(X, y)
            classifier.save_model(MODEL_PATH)
        
        # Analyze product reviews
        print("\nAnalyzing product reviews...")
        analyzer = ReviewAnalyzer(classifier)
        reviews_df = pd.read_csv(REVIEWS_DATA_PATH)
        
        results = analyzer.analyze_reviews(reviews_df, ASPECT_FILTERS)
        
        # Display results
        print("\nAnalysis Results:")
        print(f"Overall Rating: {results['overall']['rating']:.1f}/5 (from {results['overall']['prediction_count']} reviews)")
        
        if 'camera' in results:
            cam = results['camera']
            print(f"Camera Rating: {cam['rating']:.1f}/5 (from {cam['prediction_count']} mentions)")
    
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
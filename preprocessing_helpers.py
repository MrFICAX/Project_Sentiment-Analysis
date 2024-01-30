import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer


nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()
lemma = WordNetLemmatizer()


def Encode_rating(rating: int) -> str:
    if rating >= 4:
        return "Positive"
    elif rating <= 2:
        return "Negative"
    else:
        return "Neutral"
    
def Removing_url(review: str) -> str:
    return re.sub(r'http\S+', '', review)

def Clean_non_alphanumeric(review: str) -> str:
    return re.sub('[^a-zA-Z]', ' ', review)

def Convert_to_lowercase(review: str) -> str:
    return str(review).lower()

def Tokenize_text(review_text: str) -> str:
    return word_tokenize(review_text)

def Remove_stopwords(token: str) -> str:
    return [item for item in token if item not in stop_words]

def Stemming(review_text: str) -> str:
    return [stemmer.stem(token) for token in review_text]

def Lemmatization(review_text: str) -> str:
    return [lemma.lemmatize(word = token, pos = 'v') for token in review_text]

def Remove_short_words(review_text: str) -> str:
    return [token for token in review_text if len(token) > 2]

def Convert_list_of_tokens_to_string(reviews_list: list) -> str:
    return ' '.join(reviews_list)
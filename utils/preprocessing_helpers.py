import re
from typing import List, Any

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

def LabelEncode_rating(rating: int) -> int:
    if rating >= 4:
        return 2
    elif rating <= 2:
        return 0
    else:
        return 1


def Removing_url(review: str) -> str:
    return re.sub(r'http\S+', '', review)


def Clean_non_alphanumeric(review: str) -> str:
    return re.sub('[^a-zA-Z]', ' ', review)


def Convert_to_lowercase(review: str) -> str:
    return str(review).lower()


def Tokenize_text(review_text: str) -> str:
    return word_tokenize(review_text)


def Remove_stopwords(token: str) -> list[str]:
    return [item for item in token if item not in stop_words]


def Stemming(review_text: str) -> list[Any]:
    return [stemmer.stem(token) for token in review_text]


def Lemmatization(review_text: str) -> list[str]:
    return [lemma.lemmatize(word=token, pos='v') for token in review_text]


def Remove_short_words(review_text: str) -> list[str]:
    return [token for token in review_text if len(token) > 2]


def Convert_list_of_tokens_to_string(reviews_list: list) -> str:
    return ' '.join(reviews_list)


import numpy as np
import matplotlib.pyplot as plt


def get_num_words_per_sample(sample_texts):
    """Returns the median number of words per sample given corpus.

    # Arguments
        sample_texts: list, sample texts.

    # Returns
        int, median number of words per sample.
    """
    num_words = [len(s.split()) for s in sample_texts]
    plt.hist(num_words)
    plt.xlabel('Length of a sample')
    plt.ylabel('Number of samples')
    plt.title('Sample length distribution')
    plt.show()

    # return np.median(num_words)





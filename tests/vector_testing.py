# import pandas as pd
# from utils import preprocessing_helpers as preprocessing_helpers
# from utils.plotting import plot_history
# from utils.get_model import train_ngram_model
#
# dataset = pd.read_csv("../data/Amazon_Unlocked_Mobile.csv", index_col=False)
#
# dataset = dataset[['Rating', 'Reviews']]
# dataset.dropna(inplace=True)
#
# dataset['Label'] = dataset['Rating'].apply(preprocessing_helpers.LabelEncode_rating)
#
# dataset['CleanReviews'] = dataset['Reviews'].apply(preprocessing_helpers.Removing_url)
#
# dataset['CleanReviews'] = dataset['CleanReviews'].apply(preprocessing_helpers.Convert_to_lowercase)
#
# dataset['CleanReviews'] = dataset['CleanReviews'].apply(preprocessing_helpers.Clean_non_alphanumeric)
#
# dataset_filtered = dataset[['Label', 'CleanReviews']]
#
# train_size = 0.8
# train_data = dataset_filtered.loc[: len(dataset_filtered) * train_size, :]
# test_data = dataset_filtered.loc[len(dataset_filtered) * train_size:, :]
#
# data = [(train_data.loc[:, 'CleanReviews'].values, train_data.loc[:, 'Label'].values),
#           (test_data.loc[:, 'CleanReviews'].values, test_data.loc[:, 'Label'].values)]
#
# print()
import os
import random

import numpy as np


def load_imdb_sentiment_analysis_dataset(data_path, seed=123):
    """Loads the Imdb movie reviews sentiment analysis dataset.

    # Arguments
        data_path: string, path to the data directory.
        seed: int, seed for randomizer.

    # Returns
        A tuple of training and validation data.
        Number of training samples: 25000
        Number of test samples: 25000
        Number of categories: 2 (0 - negative, 1 - positive)

    # References
        Mass et al., http://www.aclweb.org/anthology/P11-1015

        Download and uncompress archive from:
        http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz
    """
    imdb_data_path = os.path.join(data_path, 'aclImdb')

    # Load the training data
    train_texts = []
    train_labels = []
    for category in ['pos', 'neg']:
        train_path = os.path.join(imdb_data_path, 'train', category)
        for fname in sorted(os.listdir(train_path)):
            if fname.endswith('.txt'):
                with open(os.path.join(train_path, fname)) as f:
                    train_texts.append(f.read())
                train_labels.append(0 if category == 'neg' else 1)

    # Load the validation data.
    test_texts = []
    test_labels = []
    for category in ['pos', 'neg']:
        test_path = os.path.join(imdb_data_path, 'test', category)
        for fname in sorted(os.listdir(test_path)):
            if fname.endswith('.txt'):
                with open(os.path.join(test_path, fname)) as f:
                    test_texts.append(f.read())
                test_labels.append(0 if category == 'neg' else 1)

    # Shuffle the training data and labels.
    random.seed(seed)
    random.shuffle(train_texts)
    random.seed(seed)
    random.shuffle(train_labels)

    return ((train_texts, np.array(train_labels)),
            (test_texts, np.array(test_labels)))



data = load_imdb_sentiment_analysis_dataset(FLAGS.data_dir)

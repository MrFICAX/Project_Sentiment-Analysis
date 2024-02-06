
import pandas as pd
from utils import preprocessing_helpers as preprocessing_helpers

dataset = pd.read_csv("../data/Amazon_Unlocked_Mobile.csv", index_col=False)


dataset = dataset[['Rating', 'Reviews']]
dataset.dropna(inplace=True)

dataset['Label'] = dataset['Rating'].apply(preprocessing_helpers.Encode_rating)

dataset['CleanReviews'] = dataset['Reviews'].apply(preprocessing_helpers.Removing_url)

dataset['CleanReviews'] = dataset['CleanReviews'].apply(preprocessing_helpers.Convert_to_lowercase)

dataset['CleanReviews'] = dataset['CleanReviews'].apply(preprocessing_helpers.Clean_non_alphanumeric)


dataset_filtered = dataset[['Label', 'CleanReviews']]

# preprocessing_helpers.plot_sample_length_distribution_longer_than(dataset_filtered.loc[:, 'CleanReviews'], 2000, 50000)
preprocessing_helpers.plot_sample_length_distribution_longer_than(dataset_filtered.loc[:, 'CleanReviews'], 0, 2000)

#preprocessing_helpers.get_num_words_per_sample(dataset_filtered.loc[:, 'CleanReviews'])
print()
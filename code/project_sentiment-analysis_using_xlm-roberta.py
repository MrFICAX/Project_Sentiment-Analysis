
import pandas as pd
from utils import preprocessing_helpers as preprocessing_helpers

from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

dataset = pd.read_csv("../data/Amazon_Unlocked_Mobile.csv", index_col=False)


dataset = dataset[['Rating', 'Reviews']]
dataset.dropna(inplace=True)

dataset['Label'] = dataset['Rating'].apply(preprocessing_helpers.Encode_rating)

dataset['CleanReviews'] = dataset['Reviews'].apply(preprocessing_helpers.Removing_url)

dataset['CleanReviews'] = dataset['CleanReviews'].apply(preprocessing_helpers.Convert_to_lowercase)

dataset['CleanReviews'] = dataset['CleanReviews'].apply(preprocessing_helpers.Clean_non_alphanumeric)


dataset_filtered = dataset[['Label', 'CleanReviews']]


import torch

# check if we have cuda installed
if torch.cuda.is_available():
    # to use GPU
    device = torch.device("cuda")
    print('There are %d GPU(s) available.' % torch.cuda.device_count())
    print('GPU is:', torch.cuda.get_device_name(0))
else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")


# import packages
import spacy
from spacy.language import Language
from spacy_langdetect import LanguageDetector


# load english vocab and create pipeline
def get_lang_detector(nlp, name):
    return LanguageDetector()

nlp = spacy.load("en_core_web_sm")
Language.factory("language_detector", func=get_lang_detector)
nlp.add_pipe('language_detector', last=True)


# use created pipeline for language detect
def detect_lan(text) :
    doc = nlp(text)
    detect_language = doc._.language
    detect_language = detect_language['language']
    return(detect_language)


# load tokens
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained('xlm-roberta-base')


# tokenize the text feature
tokenized_feature_raw = tokenizer.batch_encode_plus(
                            # Sentences to encode
                            dataset_filtered.CleanReviews.values.tolist(),
                            # Add '[CLS]' and '[SEP]'
                            add_special_tokens = True
                   )

# identify features and target
features = dataset_filtered.CleanReviews.values.tolist()
target = dataset_filtered.Label.values.tolist()

# tokenize features
MAX_LEN = 128
tokenized_feature = tokenizer.batch_encode_plus(
                            # Sentences to encode
                            features,
                            # Add '[CLS]' and '[SEP]'
                            add_special_tokens = True,
                            # Add empty tokens if len(text)<MAX_LEN
                            padding = 'max_length',
                            # Truncate all sentences to max length
                            truncation=True,
                            # Set the maximum length
                            max_length = MAX_LEN,
                            # Return attention mask
                            return_attention_mask = True,
                            # Return pytorch tensors
                            return_tensors = 'pt'
                   )


# convert label into numeric
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
le.fit(target)
target_num = le.transform(target)


# Use 80% for training and 20% for validation
from sklearn.model_selection import train_test_split

train_inputs, validation_inputs, train_labels, validation_labels, train_masks, validation_masks = train_test_split(tokenized_feature['input_ids'],
                                                                                                             target_num,
                                                                                                                    tokenized_feature['attention_mask'],
                                                                                                      random_state=2018, test_size=0.2, stratify=target)

# define batch_size
batch_size = 16
# Create the DataLoader for our training set
train_data = TensorDataset(train_inputs, train_masks, torch.tensor(train_labels))
train_sampler = RandomSampler(train_data)
train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

# Create the DataLoader for our test set
validation_data = TensorDataset(validation_inputs, validation_masks, torch.tensor(validation_labels))
validation_sampler = SequentialSampler(validation_data)
validation_dataloader = DataLoader(validation_data, sampler=validation_sampler, batch_size=batch_size)

# BertForSequenceClassification
from transformers import XLMRobertaForSequenceClassification, AdamW, get_linear_schedule_with_warmup

model = XLMRobertaForSequenceClassification.from_pretrained(
    "xlm-roberta-base",
    # Specify number of classes
    num_labels = len(set(target)),
    # Whether the model returns attentions weights
    output_attentions = False,
    # Whether the model returns all hidden-states
    output_hidden_states = False
)

# Receive the full size of the new word
model.resize_token_embeddings(len(tokenizer))

# Optimizer & Learning Rate Scheduler
optimizer = AdamW(model.parameters(),
                  lr = 2e-5,
                  eps = 1e-8
                )


# Number of training epochs
epochs = 4
# Total number of training steps is number of batches * number of epochs.
total_steps = len(train_dataloader) * epochs
# Create the learning rate scheduler
scheduler = get_linear_schedule_with_warmup(optimizer,
                                            num_warmup_steps = 0,
                                            num_training_steps = total_steps)

# Training
import time

# Store the average loss after each epoch
loss_values = []
# number of total steps for each epoch
print('total steps per epoch: ', len(train_dataloader) / batch_size)
# looping over epochs
for epoch_i in range(0, epochs):

    print('training on epoch: ', epoch_i)
    # set start time
    t0 = time.time()
    # reset total loss
    total_loss = 0
    # model in training
    model.train()
    # loop through batch
    for step, batch in enumerate(train_dataloader):
        # Progress update every 50 step
        if step % 50 == 0 and not step == 0:
            print('training on step: ', step)
            print('total time used is: {0:.2f} s'.format(time.time() - t0))
        # load data from dataloader
        b_input_ids = batch[0].to(device)
        b_input_mask = batch[1].to(device)
        b_labels = batch[2].to(device)
        # clear any previously calculated gradients
        model.zero_grad()
        # get outputs
        outputs = model(b_input_ids,
                        token_type_ids=None,
                        attention_mask=b_input_mask,
                        labels=b_labels)
        # get loss
        loss = outputs[0]
        # total loss
        total_loss += loss.item()
        # clip the norm of the gradients to 1.0.
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        # update optimizer
        optimizer.step()
        # update learning rate
        scheduler.step()
    # Calculate the average loss over the training data.
    avg_train_loss = total_loss / len(train_dataloader)
    # Store the loss value for plotting the learning curve.
    loss_values.append(avg_train_loss)
    print("average training loss: {0:.2f}".format(avg_train_loss))
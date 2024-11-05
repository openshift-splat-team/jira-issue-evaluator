from __future__ import annotations

import nltk
import time
import keras
from typing import cast
from jira import JIRA
from jira.client import ResultList
from jira.resources import Issue
import os
import random
import sys
import warnings
warnings.filterwarnings('ignore')

# libraries for data visualization and processing
import pandas as pd
import numpy as np
np.set_printoptions(threshold=sys.maxsize)

# Importing all the required sub-modules from sklearn to enable model analysis.

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

# Importing all the required sub-modules from Keras
from keras.models import Sequential, Model
from keras.applications.vgg16 import VGG16
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, BatchNormalization, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

import re
import numpy as np
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

# Libraries for interacting with tensorflow
os.environ["TF_MIN_GPU_MULTIPROCESSOR_COUNT"] = "6"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
tf.get_logger().setLevel('ERROR')
from tensorflow.keras import backend
from tensorflow.keras import layers

nltk.download('stopwords')
nltk.download('wordnet')

poll_rate_minutes = 60
if "POLL_RATE" in os.environ:
    poll_rate_minutes = int(os.environ["POLL_RATE"])

def initializeBackend():
    """
    reinitializes keras and configures random seeds to ensure training and evaluation consistency.
    """
    backend.clear_session()
    np.random.seed(42)
    random.seed(42)
    tf.random.set_seed(42)
    print("** cleared keras backend session and set random seeds")

lemmatizer = WordNetLemmatizer()

def clean_data(review):    
    review = re.sub('[^a-zA-Z]', ' ',review)
    review = review.lower()
    return review

def remove_stop_words(review):
    review_minus_sw = []
    stop_words = stopwords.words('english')
    review = review.split()
    review = [review_minus_sw.append(word) for word in review if word not in stop_words]
    review = ' '.join(review_minus_sw)
    return review

def lematize(review):
    review = review.split()
    review = [lemmatizer.lemmatize(w) for w in review]
    review = ' '.join(review)
    return review

def process_issue(issue):
    context = issue.fields.summary + " " + issue.fields.description
    context = clean_data(context)
    context = clean_data(context)

def vectorize_text(text ,label):  
  text = tf.expand_dims(text, -1)  
  return vectorize_layer(text), label

def custom_standardization(input_data):
    lowercase = tf.strings.lower(input_data)
    input_data = lematize(input_data)
    input_data = clean_data(input_data)
    input_data = remove_stop_words(input_data)
    return input_data

df_org = pd.read_csv("/training/training.csv")

df = df_org[['Summary', 'Description', 'REFINED']]
df['issue_context'] = df['Description'].str.cat(df['Summary'], sep=' ')
df = df[df['issue_context'].apply(lambda x: isinstance(x, str))]

df['issue_context'] = df['issue_context'].apply(clean_data)
df['issue_context'] = df['issue_context'].apply(remove_stop_words)
df['issue_context'] = df['issue_context'].apply(lematize)

df.drop(["Summary", "Description"], axis=1, inplace=True)

df['REFINED'] = df['REFINED'].map({'Y': 1, 'N': 0})

def splitDataframeToDirectory(path, dataframe):
    idx = 0
    for row in df.itertuples():     
        class_path = "refined"
        if row.REFINED == 0:
            class_path = 'unrefined'
        with open(path + class_path + "/" + str(idx) + ".txt", "w") as file:
            file.write(row.issue_context)
        idx += 1

def isTypeOfInterest(issue):
    issue_type = issue.raw["fields"]["issuetype"]["name"].lower()
    return  issue_type == "sub-task" or issue_type == "task" or issue_type == "issue" or issue_type == "spike"

def isReadyForRefinement(issue):
    if not isTypeOfInterest(issue):
        return False, 0.0
        
    labels = issue.raw["fields"]["labels"]
    
    if len(labels) == 0:
        return False, 0.0
        
    for label in labels:
        if label == "ready-for-prioritization" or \
            label == "splat-bot-disable-automation":
            return False, 0.0

    if type(issue.fields.description) == type(None) or \
        type(issue.fields.summary) == type(None):
            return False, 0.0
    
    context = issue.fields.summary + " " + issue.fields.description    
    context = custom_standardization(context)
    vectorized, _ = vectorize_text(context, "")
    result = model.predict(vectorized)
    
    # if the issue has a high probability of being refined, return true. otherwise, leave it as-is.
    prob = result.ravel()[0]
    return result.ravel()[0] < 0.50, prob

SIZING_FIELD_1 = "customfield_12314040"
SIZING_FIELD_2 = "customfield_12310243"

def syncStoryPoints(issues):
    print("syncing story points")

    for issue in issues:
        val1 = issue.raw["fields"][SIZING_FIELD_1]
        val2 = issue.raw["fields"][SIZING_FIELD_2]
        
        if val1 == None and val2 != None:
            print("syncing", issue.key)
            issue.update(fields={SIZING_FIELD_1: val2})
        elif val2 == None and val1 != None:
            print("syncing", issue.key)
            issue.update(fields={SIZING_FIELD_2: val1})

splitDataframeToDirectory("refinement/training/", df)

batch_size = 32
seed = 42

raw_train_ds = tf.keras.utils.text_dataset_from_directory(
    'refinement/training',
    batch_size=batch_size,
    validation_split=0.2,
    shuffle=True,
    subset='training',
    seed=seed)

raw_val_ds = tf.keras.utils.text_dataset_from_directory(
    'refinement/training',
    batch_size=batch_size,
    validation_split=0.2,
    shuffle=True,
    subset='validation',
    seed=seed)

max_features = 10000
sequence_length = 250
embedding_dim = 128

vectorize_layer = layers.TextVectorization(
    max_tokens=max_features,
    output_mode='int',
    output_sequence_length=sequence_length)

train_text = raw_train_ds.map(lambda x, y: x)
vectorize_layer.adapt(train_text)

train_ds = raw_train_ds.map(vectorize_text)
val_ds = raw_val_ds.map(vectorize_text)

initializeBackend()
inputs = keras.Input(shape=(None,), dtype="int64")

# Next, we add a layer to map those vocab indices into a space of dimensionality
# 'embedding_dim'.
x = layers.Embedding(max_features, embedding_dim)(inputs)
x = layers.Dropout(0.5)(x)

# Conv1D + global max pooling
x = layers.Conv1D(128, 7, padding="valid", activation="relu", strides=3)(x)
x = layers.Conv1D(128, 7, padding="valid", activation="relu", strides=3)(x)
x = layers.Bidirectional(layers.LSTM(32, return_sequences=True))(x)
x = layers.GlobalMaxPooling1D()(x)

# We add a vanilla hidden layer:
x = layers.Dense(128, activation="relu")(x)
x = layers.Dropout(0.5)(x)

# We project onto a single unit output layer, and squash it with a sigmoid:
predictions = layers.Dense(1, activation="sigmoid", name="predictions")(x)

model = keras.Model(inputs, predictions)

opt = keras.optimizers.Adamax(learning_rate=0.0001)
# Compile the model with binary crossentropy loss and an adam optimizer.
model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])

epochs = 100

callbacks = []
print("** ReduceLROnPlateau callback is enabled after the first 5 epochs. Patience is 3.")
callbacks.append(
    ReduceLROnPlateau(
        monitor='val_loss',
        patience=3,
        verbose=1,
        restore_best_weights=True,
        start_from_epoch=5)
)

model_history = model.fit(train_ds, validation_data=val_ds, epochs=epochs, callbacks=callbacks)

jira_token = os.environ["JIRA_PERSONAL_ACCESS_TOKEN"]
jira_url = os.environ["JIRA_URL"]

jira = JIRA(server=jira_url,
           token_auth=jira_token)

while True:
    issues = cast(ResultList[Issue], jira.search_issues(jql_str='project in (SPLAT, OPCT) AND created > -26w AND status not in (Closed) AND creator not in (cucushift-bot, sgaoshang, "yanhli@redhat.com") AND issuetype not in (Epic) AND ("Story Points" is not EMPTY OR cf[12314040] is not EMPTY)'))
    syncStoryPoints(issues)

    issues = cast(ResultList[Issue], jira.search_issues(jql_str='project in (SPLAT, OPCT) AND created > -26w AND status not in (Closed) AND creator not in (cucushift-bot, sgaoshang, "yanhli@redhat.com") AND issuetype not in (Epic) AND (labels in (needs-refinement) OR priority = Undefined or "Story Points" IS EMPTY ) AND status  != Closed ORDER BY key DESC, priority ASC, created ASC, Rank ASC'))
    for issuekey in issues:
        issue = jira.issue(issuekey)
        ready, prob = isReadyForRefinement(issue)
        print("issue", issuekey, "refinement confidence", prob)
        if ready:        
            print("marking issue ", issuekey, " as ready-for-prioritization")        
            jira.add_comment(issue, 
                            "splat-jira-bot believes this issue is ready for refinement with a probability of " + str(prob) + ". \
                            Probabilities are expressed as a float of value [0..1] where 0 is confidence of readiness. Issues with a probability of < 0.50 are marked ready-for-prioritization." \
                            " If the issue is not ready for refinement, please set the labels `needs-refinement` and `splat-bot-disable-automation` and this issue will be reviewed to understand the " \
                            "descrepancy.", 
                            visibility={"type": "group", "value": "jira-users"})
            new_labels = []        
            labels = issue.get_field("labels")
            for label in labels:
                if label == "needs-refinement":
                    continue
                new_labels.append(label)
            new_labels.append("ready-for-prioritization")
            labels.clear()
            for new_label in new_labels:
                labels.append(new_label)
            issue.update(fields={"labels": labels})
    time.sleep(60 * poll_rate_minutes)

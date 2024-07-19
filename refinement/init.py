import random
import pandas as pd
import numpy as np
import json
import spacy
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.utils import to_categorical
from sklearn.feature_extraction.text import TfidfVectorizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras import backend
import tensorflow as tf
from tensorflow.keras.optimizers import Adamax



nlp = spacy.load("en_core_web_sm")

class_names_list = ['0', '1']

def metrics_score(actual, predicted):

    from sklearn.metrics import classification_report

    from sklearn.metrics import confusion_matrix

    print(classification_report(actual, predicted))

    print(confusion_matrix(actual, predicted))

def preprocess_text(text):
    doc = nlp(text)
    tokens = [token.lemma_.lower() for token in doc if not token.is_punct and not token.is_stop]
    return ' '.join(tokens)

def getVectorized(col, df_in):
    # Initialize the TfidfVectorizer
    vectorizer = TfidfVectorizer()

    # Fit the vectorizer on the entire text data
    vectorizer.fit(df_in[col])

    # Transform the entire text data
    tfidf_matrix = vectorizer.transform(df_in[col])

    # Convert the sparse matrix to a dense format
    tfidf_dense = tfidf_matrix.toarray()

    # Get the feature names (vocabulary)
    feature_names = vectorizer.get_feature_names_out()

    # Create a new DataFrame with the TF-IDF features
    tfidf_df = pd.DataFrame(tfidf_dense, columns=feature_names)

    return tfidf_df


def prepareDataframe(df_in):
    df_work = df_in[["Description", "Summary", "REFINED"]]    
    df_work["Description"] = df_work["Description"].fillna("")
    df_work['issue_text'] = df_work["Summary"] + " " + df_work["Description"]
    df_vect = getVectorized("issue_text", df_work)        
    df_out = pd.concat([df_work["REFINED"], df_vect], axis=1)
    return df_out

def main():
    df = pd.read_csv("/training/training.csv")

    df_full = prepareDataframe(df)
    mapping = {'Y': 1, 'N': 0}
    df_full['REFINED'] = df_full['REFINED'].map(mapping)

    x_cols = df_full.columns[df_full.columns != "REFINED"]

    x = df_full[x_cols].values
    y = df_full["REFINED"].values

    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, shuffle=True, random_state=42)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    backend.clear_session()

    random.seed(42)
    np.random.seed(42)
    tf.random.set_seed(42)


    model = Sequential()
    model.add(Dense(1024, input_dim=X_train.shape[1], activation='relu'))
    model.add(Dense(1024, activation='relu'))
    # model.add(Dropout(0.3))
    # model.add(Dense(64, activation='relu'))
    # model.add(Dropout(0.2))
    model.add(Dense(1, activation='sigmoid'))

    opt = Adamax(learning_rate=0.00001)

    model.compile(optimizer=opt, loss='binary_focal_crossentropy', metrics=['accuracy'])  # Change loss function for regression/multi-class

    # Train the model
    training_stats = model.fit(X_train, y_train, epochs=40, batch_size=50, validation_data=(X_test, y_test), verbose=2)

    model.summary()

    model.save("jira_sizing.keras")

    y_pred_prob = model.predict(X_test)
    y_pred = (y_pred_prob > 0.5).astype(int)
    metrics_score(y_test, y_pred)

    file_path = 'features.json'    
    with open(file_path, 'w') as file:
        features = []
        for feature in df_full.columns:
            features.append(feature)
        json.dump(features, file, indent=4)
    return scaler

if __name__ == "__main__":
   # stuff only to run when not called via 'import' here
   main()
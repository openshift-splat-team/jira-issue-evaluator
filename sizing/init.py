import random
import pandas as pd
import numpy as np
import json
import spacy
from joblib import dump
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.utils import to_categorical
from sklearn.feature_extraction.text import TfidfVectorizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras import backend
import tensorflow as tf
from tensorflow.keras.optimizers import Adam, Ftrl, Adamax



nlp = spacy.load("en_core_web_sm")

def metrics_score(actual, predicted):

    from sklearn.metrics import classification_report

    from sklearn.metrics import confusion_matrix

    print(classification_report(actual, predicted))

    print(confusion_matrix(actual, predicted))


def preprocess_text(text):
    doc = nlp(text)
    tokens = [token.lemma_.lower() for token in doc if not token.is_punct and not token.is_stop]
    return ' '.join(tokens)

#def getVectorized(col, df_in):
#    vectorizer = TfidfVectorizer()        
#    matrix = vectorizer.fit_transform(df_in[col].apply(preprocess_text))    
#    return pd.DataFrame(matrix.toarray(), columns=vectorizer.get_feature_names_out())

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
    df_work = df_in[["Description", "Summary", "Issue Type", "Custom field (Story Points)"]]
    df_work["Description"] = df_work["Description"].fillna("")
    #df_work['issue_text'] = df_work["Summary"] + " " + df_work["Description"]
    df_work.loc[:, 'issue_text'] = df_work['Summary'] + " " + df_work['Description']

    df_vect = getVectorized("issue_text", df_work)

    df_cat = pd.get_dummies(df_work, columns=['Issue Type'])            
    df_cat.drop(["Description", "Summary", "issue_text"], axis=1, inplace=True)         
    df_out = pd.concat([df_cat, df_vect], axis=1)    
    return df_out

def main():
    df = pd.read_csv("/training/training.csv")

    df_full = prepareDataframe(df)

    x_cols = df_full.columns[df_full.columns != "Custom field (Story Points)"]

    x = df_full[x_cols].values
    df_full["Custom field (Story Points)"] = df_full["Custom field (Story Points)"].astype("float")
    y = df_full["Custom field (Story Points)"].values
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, shuffle=True, random_state=42)

    num_classes = 14
    y_train_encoded = to_categorical(y_train, num_classes=num_classes)
    y_test_encoded = to_categorical(y_test, num_classes=num_classes)

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
    model.add(Dense(14, activation='softmax'))

    opt = Adamax(learning_rate=0.00001)
    
    model.compile(optimizer=opt, loss='categorical_focal_crossentropy', metrics=['accuracy'])  # Change loss function for regression/multi-class

    # Train the model
    training_stats = model.fit(X_train, y_train_encoded, epochs=100, batch_size=32, validation_data=(X_test, y_test_encoded))

    model.summary()

    model.save("jira_sizing.keras")
    
    y_pred = model.predict(X_test)

    predict_class = np.argmax(y_pred, axis=1)
    predict_class = predict_class.tolist()

    file_path = 'features.json'    
    with open(file_path, 'w') as file:
        features = []
        for feature in df_full.columns:
            features.append(feature)
        json.dump(features, file, indent=4)
    
    metrics_score(y_test, predict_class)
    
    return scaler

if __name__ == "__main__":
   # stuff only to run when not called via 'import' here
   main()
import json
from http.server import SimpleHTTPRequestHandler, HTTPServer
import gzip
import os
import torch
import string
import numpy as np
from tensorflow.keras.models import Sequential
from keras.models import load_model
from sklearn.preprocessing import StandardScaler
import pandas as pd
from init import prepareDataframe, main

class QueryHandler(SimpleHTTPRequestHandler):
    def do_POST(self):        
        length = int(self.headers['Content-Length'])
        messagecontent = self.rfile.read(length)
        response = performSizing(content = str(messagecontent, "utf-8"))                
        self.send_response(200)
        self.send_header('Content-Type', 'application/json; charset=utf-8')        
        self.end_headers()
        self.wfile.write(response.encode('utf-8'))

def run(server_class=HTTPServer, handler_class=QueryHandler, port=8002):
    server_address = ('', port)
    httpd = server_class(server_address, handler_class)
    print(f'Starting httpd server on port {port}...')
    httpd.serve_forever()

def performSizing(content):    
    with open("features.json", 'r') as featuresJson:
        features = json.load(featuresJson)
        df_in = pd.DataFrame(json.loads(content))
        df_in["REFINED"] = 0
        df = prepareDataframe(df_in)
        missing_features = [feature for feature in features if feature not in df.columns]
        missing_data = pd.DataFrame(0, index=df.index, columns=missing_features)
        new_data_complete = pd.concat([df, missing_data], axis=1)
        new_data_complete = new_data_complete[features]
        new_data_complete.drop(["REFINED"], axis=1, inplace=True)
        #scaler = StandardScaler()
        #scaler.fit(new_data_complete)
        new_data_scaled = scaler.transform(new_data_complete)
        predictions = model.predict(new_data_scaled)
        
        scores = []
        confidence = []
        for prediction in predictions:
            confidence.append(prediction)
            if prediction > 0.5:
                scores.append("true")
            else:
                scores.append("false")
        
        confidence_df = pd.DataFrame(confidence, columns=["confidence"])
        score_df = pd.DataFrame(scores, columns=["REFINED"])
        minimized_df = df_in["Issue Key"]
        
        return pd.concat([minimized_df, score_df, confidence_df], axis=1).to_json()        

# initialize the datamodel 
scaler = main()
model = load_model("jira_sizing.keras")
run()
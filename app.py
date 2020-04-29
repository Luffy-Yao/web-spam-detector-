#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  6 12:39:42 2020

@author: linxing
"""
import pandas as pd
from flask import Flask, request, jsonify
import numpy as np
import mmh3
from nltk.util import ngrams
import pickle as p

def extract_tokens(element):
    tokens=str(element).rsplit("/")
    return tokens

def extract_host(element):
    host=str(element).rsplit(".")
    return host


def eng_hash_test(data, vdim=1000):
    ## take 3 n-gram of the url and hash it into a vector of length 1000
        final = []
        v = [0] * vdim
        new = list(ngrams(data, 3))
        for i in new:
            new_ = ''.join(i)
            idx = mmh3.hash(new_) % vdim
            v[idx] += 1
        final.append([np.array(v)])
        return final
    

# load model
model = p.load(open('model.pkl','rb'))

# app
app = Flask(__name__)

# routes
@app.route('/', methods=['POST'])

def predict():
    if request.method=='POST':
       # get data
        data = request.get_json(force=True) 
       # convert data into dataframe
        data.update((x, [y]) for x, y in data.items())
        data_df = pd.DataFrame.from_dict(data)
      # retrieve host, url
        host,url=(data_df['host'],data_df['url'])
      # transform host,url into a vector with right dimension
        temp=extract_tokens(url)+extract_host(host)[::-1]
        temp_data=eng_hash_test(temp)
        temp_data=np.array(temp_data)
        temp_data=temp_data.reshape(temp_data.shape[0],temp_data.shape[2])
      # predict the class of input text
        result=model.predict_classes(temp_data,verbose=0)
      # send back to the browser
        output = {'results': int(result[0])}
    # return data
    return jsonify(results=output)

if __name__ == '__main__':
    app.run(port = 1234, debug=True)
    


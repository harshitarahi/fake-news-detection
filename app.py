# -*- coding: utf-8 -*-
"""
Created on Wed Mar 17 12:04:32 2021

@author: H&A
"""
import numpy as np
from flask import Flask, request, jsonify, render_template
from User import *

app = Flask(__name__)
model = compile_model()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    #int_features = [int(x) for x in request.form.values()]
    #final_features = [np.array(int_features)]
    #prediction = model.predict(final_features)

    #output = round(prediction[0], 2)
    
    
    news = request.form.values()
    padded_docs = newsPredict(news)
     
    output = np.argmax(model.predict(padded_docs), axis=-1)
    output_list = ['barely-true', 'false', 'half-true', 'mostly-true', 'pants-fire', 'true']
    
    
    
    return render_template('index.html', prediction_text='This news seems to be  $ {}'.format(output_list[output]))


if __name__ == "__main__":
    app.run(debug=True)
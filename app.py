import numpy as np
from flask import Flask, request, jsonify, render_template
from User import *

app = Flask(__name__)
model = compile_model()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/fetch')
def fetch():
    global model
    model = load_model()
    return render_template('index.html')
    
@app.route('/predict',methods=['POST'])
def predict():
    news = request.form["News Title"]
    padded_docs = newsPredict(news)
     
    output = np.argmax(model.predict(padded_docs), axis=-1)
    output_list = ['barely-true', 'true', 'half-true', 'mostly-true', 'pants-fire', 'false']
    
    return render_template('index.html', prediction_text='This news seems to be  $ {}'.format(output_list[output[0]]))


if __name__ == "__main__":
    app.run(debug=True)

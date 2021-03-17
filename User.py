# -*- coding: utf-8 -*-
"""
Created on Tue Mar 16 19:24:21 2021

@author: Anshul Mudliar
"""

import numpy as np
from nltk.stem.porter import PorterStemmer
import re
import tensorflow as tf
from tensorflow.keras.layers import Embedding
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import one_hot
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import *
from tensorflow.keras import layers
from tensorflow.keras import regularizers
from tensorflow import keras

from nltk.corpus import stopwords
import nltk
nltk.download('stopwords')


vocab_size = 10000
max_length = 50

def newsPredict(news):
    
    
    
    ps = PorterStemmer()
    latestNews = news
    
    corpus_latest = []
    
    review = re.sub('[^a-zA-Z]', ' ', latestNews)
    review = review.lower()
    review = review.split()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    corpus_latest.append(review)
    
    encoded_docs_latest = [one_hot(d, vocab_size) for d in corpus_latest]
    padded_docs_latest = pad_sequences(encoded_docs_latest, maxlen=max_length, padding='post')
    
    return padded_docs_latest



def get_optimizer():
    
  lr_schedule = tf.keras.optimizers.schedules.InverseTimeDecay(
                0.001,
                decay_steps=100,
                decay_rate=1,
                staircase=False)
  return tf.keras.optimizers.Adam(lr_schedule)

    
def compile_model():
    
    model = keras.models.load_model('tryNew/')
     
    return model

def predict_class():
    
     news = input("Enter the news")
     
     model = compile_model()
     padded_docs = newsPredict(news)
     
     print(np.argmax(model.predict(padded_docs), axis=-1))
    
compile_model()
    
    

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


vocab_size = 10000
max_length = 50
stopword = {'a','about','above','after','again','against','ain','all','am','an','and','any', 'are', 
            'aren', "aren't", 'as', 'at', 'be', 'because', 'been', 'before', 'being', 'below', 
            'between', 'both', 'but', 'by', 'can', 'couldn', "couldn't",'d', 'did', 'didn', "didn't", 
            'do', 'does', 'doesn', "doesn't", 'doing', 'don', "don't", 'down', 'during', 'each', 'few', 
            'for', 'from', 'further', 'had', 'hadn', "hadn't", 'has', 'hasn', "hasn't", 'have', 'haven', 
            "haven't", 'having', 'he', 'her', 'here', 'hers', 'herself', 'him', 'himself', 'his', 'how', 
            'i', 'if', 'in', 'into', 'is', 'isn', "isn't", 'it', "it's", 'its', 'itself', 'just', 'll', 
            'm', 'ma', 'me', 'mightn', "mightn't", 'more', 'most', 'mustn', "mustn't", 'my', 'myself', 
            'needn', "needn't", 'no', 'nor','not','now', 'o', 'of', 'off', 'on', 'once', 'only', 'or', 
            'other', 'our', 'ours', 'ourselves', 'out', 'over', 'own', 're', 's', 'same', 'shan', "shan't", 
            'she', "she's", 'should', "should've", 'shouldn', "shouldn't", 'so', 'some', 'such', 't',
            'than', 'that', "that'll", 'the', 'their', 'theirs', 'them', 'themselves', 'then', 'there',
            'these', 'they', 'this', 'those', 'through', 'to', 'too', 'under', 'until', 'up', 've', 
            'very', 'was', 'wasn', "wasn't", 'we', 'were', 'weren', "weren't", 'what', 'when', 'where', 
            'which', 'while', 'who', 'whom', 'why', 'will', 'with', 'won', "won't", 'wouldn', "wouldn't", 
            'y', 'you', "you'd", "you'll", "you're", "you've", 'your', 'yours', 'yourself', 'yourselves'}

def newsPredict(news):
    
    
    
    ps = PorterStemmer()
    latestNews = news
    
    corpus_latest = []
    
    review = re.sub('[^a-zA-Z]', ' ', latestNews)
    review = review.lower()
    review = review.split()
    review = [ps.stem(word) for word in review if not word in stopword]
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

   
    
    

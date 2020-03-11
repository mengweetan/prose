import tensorflow as tf

import numpy as np
import pandas as pd

Model = tf.keras.models.Model
Input = tf.keras.layers.Input
LSTM = tf.keras.layers.LSTM
Embedding = tf.keras.layers.Embedding
Dense = tf.keras.layers.Dense
concatenate = tf.keras.layers.concatenate
Flatten = tf.keras.layers.Flatten
Add =  tf.keras.layers.Add

pad_sequences = tf.keras.preprocessing.sequence.pad_sequences
Tokenizer = tf.keras.preprocessing.text.Tokenizer

dataDir='../data/'
HAIKU_LINES_NUM = 3

df = pd.read_csv(dataDir+'__INPUT.txt', sep = '\t')
df.info()

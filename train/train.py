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

df = df[:100]
t = Tokenizer()
t.fit_on_texts([df['input_texts'][i] for i in range(df.shape[0])])
vocab_size = len(t.word_index) + 1 # note - padded 1

encoded_docs = t.texts_to_sequences([df['input_texts'][i] for i in range(df.shape[0])])
max_len = max([len(l) for l in encoded_docs]) # get the max len
padded_docs = pad_sequences(encoded_docs, maxlen=max_len, padding='post')

'''

# do this locally!

embeddings_index = dict()
f = open(dataDir+'glove.6B/glove.6B.100d.txt') # try 100 dimension
for line in f:
	values = line.split()
	word = values[0]
	coefs = np.asarray(values[1:], dtype='float32')
	embeddings_index[word] = coefs
f.close()
print('Loaded %s word vectors.' % len(embeddings_index))

embedding_matrix = np.zeros((vocab_size, 100)) # because we are using 100 dimension pre trained embedding
for word, i in t.word_index.items():
	embedding_vector = embeddings_index.get(word)
	if embedding_vector is not None:
		embedding_matrix[i] = embedding_vector

print (embedding_matrix)
print (type(embedding_matrix))
print (embedding_matrix.shape)


from numpy import savetxt, loadtxt
savetxt(dataDir+'embedding_matrix.csv', embedding_matrix, delimiter=',')
'''

embedding_matrix = np.loadtxt(open(dataDir+'embedding_matrix.csv', "rb"), delimiter=",")
print (type(embedding_matrix))
print (embedding_matrix.shape)

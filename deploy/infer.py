import tensorflow as tf

import numpy as np
import pandas as pd
import os

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

seed ='some random keywords'
#dataDir='./data/'
dataDir='data/'


from argparse import ArgumentParser

parser = ArgumentParser()

parser.add_argument("-s", "--seed",dest="seed",
                    help="seed keywords"
                    )

seed = parser.parse_args().seed






t = Tokenizer()
t.fit_on_texts([seed])
t.word_index


encoded_docs = t.texts_to_sequences([seed])
max_len = len(encoded_docs[0])

embedding_matrix = np.loadtxt(open(dataDir+'embedding_matrix.csv', "rb"), delimiter=",")
print (embedding_matrix.shape)

latent_dim = 50 # note using 50 dim GLOVE
HAIKU_LINES_NUM = 3

load_model = tf.keras.models.load_model
#model = load_model('model/haiku/modelv2-1.h5',custom_objects={'tf': tf}, compile=False)

# TEMP CODES
num_encoder_tokens = 50000
num_decoder_tokens = 50000

inputs = Input(shape=(None,))
aux_inputs = [Input(shape=(None,), name='haikuLine_{}'.format(i)) for i in range(HAIKU_LINES_NUM)]
syllabus_inputs = [Input(shape=(1,),  dtype='int32', name='syllabus_input_{}'.format(i)) for i in range(HAIKU_LINES_NUM)]

x =  Embedding(num_encoder_tokens+1, latent_dim, mask_zero = True)(inputs)
_ , state_h, state_c = LSTM(latent_dim,  return_state=True) (x)

encoder_model = Model([inputs, syllabus_inputs[0],syllabus_inputs[1],syllabus_inputs[2]], [state_h, state_c])

'''
model = load_model('model/haiku/modelv2-2.h5')
encoder_model = Model( [ model.inputs[0], model.inputs[4], model.inputs[5], model.inputs[6]], [model.layers[4].output[1],model.layers[4].output[2] ])
'''

#decoder model

decoder_state_input_h = Input(shape=(latent_dim,))
decoder_state_input_c = Input(shape=(latent_dim,))
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]



syllabus_state_hs = []
syllabus_state_cs = []
syllabus_dense = []

last_states_hs = []
last_states_cs = []

decoder_outputs2 = []


for i in range(self.HAIKU_LINES_NUM):

    x2 = Embedding(self.num_decoder_tokens[i]+1, latent_dim,  name='s_emb{}'.format(i)) (syllabus_inputs[i])
    x2, s_state_h, s_state_c = LSTM(latent_dim , return_sequences=True,  return_state=True, name='lstm_post_emb{}'.format(i)) (x2)
    syllabus_state_hs.append(s_state_h)
    syllabus_state_cs.append(s_state_c)



    x= Embedding(self.num_decoder_tokens[i]+1, latent_dim, mask_zero = True)  (aux_inputs[i]) # Get the embeddings of the decoder sequence
    x, state_h2, state_c2 =  LSTM(latent_dim, return_sequences=True, return_state=True, name='lstm_post_line_emb{}'.format(i)) (x, initial_state=decoder_states_inputs)

    if i == 0:
        x, x_state_h, x_state_c = LSTM(latent_dim , return_sequences=True,  return_state=True, name='lstm_line{}'.format(i)) (x,   initial_state=[state_h2 + syllabus_state_hs[i], state_c2 + syllabus_state_cs[i]])
        last_states_hs.append(x_state_h)
        last_states_cs.append(x_state_c)

    else:
        x, x_state_h, x_state_c = LSTM(latent_dim , return_sequences=True,  return_state=True,name='lstm_line{}'.format(i)) (x,   initial_state=[state_h2 + last_states_hs[i-1]+syllabus_state_hs[i],state_c2 +  last_states_cs[i-1]+syllabus_state_cs[i]])
        last_states_hs.append(x_state_h)
        last_states_cs.append(x_state_c)

    #print (x.shape)
    decoder_outputs2.append(Dense(self.num_decoder_tokens[i], activation='softmax', name='predict{}'.format(i)) (x))


#########


for i in range(self.HAIKU_LINES_NUM):

    x2 = Embedding(self.num_decoder_tokens[i]+1, latent_dim,  name='syllabus{}'.format(i)) (syllabus_inputs[i])
    x2, s_state_h, s_state_c = LSTM(latent_dim , return_sequences=True,  return_state=True, name='syllabus_lstm{}'.format(i)) (x2)
    syllabus_state_hs.append(s_state_h)
    syllabus_state_cs.append(s_state_c)

    syllabus_dense.append( Dense(latent_dim , activation='softmax')  (syllabus_inputs[i]) )

    x = Embedding(self.num_decoder_tokens[i]+1, latent_dim, mask_zero = True, name='line{}'.format(i)) (aux_inputs[i])
    if i == 0:
        x, x_state_h, x_state_c = LSTM(latent_dim , return_sequences=True,  return_state=True, name='lstm{}'.format(i)) \
            (x,   initial_state=[Add()([state_h , syllabus_dense[i]]), Add()([state_c , syllabus_dense[i]])])
            #(x,   initial_state=[Add()([state_h , syllabus_state_hs[i]]), Add()([state_c , syllabus_state_cs[i]])])


        #x, x_state_h, x_state_c = LSTM(latent_dim , return_sequences=True,  return_state=True, name='lstm{}'.format(i)) (x,   initial_state=[Add([state_h , syllabus_state_hs[i]]), Add([state_c , syllabus_state_cs[i]])])

        last_states_hs.append(x_state_h)
        last_states_cs.append(x_state_c)

    else:
        #x, x_state_h, x_state_c = LSTM(latent_dim , return_sequences=True,  return_state=True,name='lstm{}'.format(i)) (x,   initial_state=[state_h + last_states_hs[i-1]+syllabus_state_hs[i], state_c + last_states_cs[i-1]+syllabus_state_cs[i]])
        x, x_state_h, x_state_c = LSTM(latent_dim , return_sequences=True,  return_state=True,name='lstm{}'.format(i)) \
            (x,   initial_state=[Add()([state_h , last_states_hs[i-1], syllabus_dense[i]]), Add()([state_c , last_states_cs[i-1], syllabus_dense[i]])])
            #(x,   initial_state=[Add()([state_h , last_states_hs[i-1], syllabus_state_hs[i]]), Add()([state_c , last_states_cs[i-1], syllabus_state_cs[i]])])

        last_states_hs.append(x_state_h)
        last_states_cs.append(x_state_c)


    outputs.append(Dense(self.num_decoder_tokens[i], activation='softmax', name='predict{}'.format(i)) (x))





#########

decoder_model = Model(
    [aux_inputs] + decoder_states_inputs + [syllabus_inputs],
    #[decoder_outputs2[0] , decoder_outputs2[1] , decoder_outputs2[2]] + [state_h2, state_c2])
    [tuple(np.array(decoder_outputs2).tolist())] + [state_h2, state_c2])
#decoder_model.load_weights(self.data_dir+'model8_weights.h5')
#decoder_model.summary()

print (seed)

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

dataDir='data/'
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
print (embedding_matrix.shape)

def _process_target_texts(text):
    if 'start__ ' not in text:
        return 'start__ '+text+' __end'

df.line1=df['line1'].apply(_process_target_texts)
df.line2=df['line2'].apply(_process_target_texts)
df.line3=df['line3'].apply(_process_target_texts)

def _arrayOfInt(texts):
    try:
        array = texts.split(',')
        array = [int(i) for i in array]
    except:
        print (texts)
    return array

df.lib = df['lib'].apply(_arrayOfInt)

targetTexts = [df.line1, df.line2, df.line3 ]
target_words = [ set(' '.join(item).split(' ')) for item in targetTexts ]

num_encoder_tokens = vocab_size
num_decoder_tokens = [len(i) for i in target_words ]

for i in range(HAIKU_LINES_NUM):
    num_decoder_tokens[i] += 1

target_token_index = [dict( [(char, i+1) for i, char in enumerate(j)]) for j in target_words]
reverse_target_char_index = [dict((i, word) for word, i in j.items()) for j in target_token_index ]

max_encoder_seq_length = max_len
max_decoder_seq_length = [ max ([len(column[i].split(' ')) for i in range(df.shape[0])]) for column in targetTexts]

input_token_index = t.word_index
reverse_input_char_index = dict(map(reversed, t.word_index.items()))

y = np.array( targetTexts)
y = y.transpose()
X = (padded_docs , df.lib)

epochs = 3
latent_dim = 100
dropout=0.1 #regularization , to prevent over fitting
learning_rate = 0.005
optimizer = 'rmsprop'
lstm_dim =latent_dim

inputs = Input(shape=(max_len,))

# from tensorflow.contrib.keras.api.keras.initializers import Constant
# n = Embedding(2, 2, embeddings_initializer=Constant(m), input_length=1, name='embedding_matrix_1', trainable=False)

Constant = tf.compat.v1.keras.initializers.Constant
x =  Embedding(num_encoder_tokens, latent_dim,  embeddings_initializer=Constant(embedding_matrix), input_length=max_len, trainable=False, mask_zero = True)(inputs)
# x =  Embedding(num_encoder_tokens, latent_dim,  weights=[embedding_matrix], input_length=max_len, trainable=False, mask_zero = True)(inputs)
_ , state_h, state_c = LSTM(latent_dim,  return_state=True) (x)


aux_inputs = [Input(shape=(None,), name='aux_input_{}'.format(i)) for i in range(HAIKU_LINES_NUM)]

syllabus_inputs = [Input(shape=(1,), name='syllabus_input_{}'.format(i)) for i in range(HAIKU_LINES_NUM)] # assume training data has 12 types of lines


syllabus_dense = []

last_states_hs = []
last_states_cs = []

outputs = []

for i in range(HAIKU_LINES_NUM):

    syllabus_dense.append( Dense(latent_dim , activation='softmax')  (syllabus_inputs[i]) )
    x = Embedding(num_decoder_tokens[i]+1, latent_dim, mask_zero = True, name='line{}'.format(i)) (aux_inputs[i])
    if i == 0:
        x, x_state_h, x_state_c = LSTM(lstm_dim , return_sequences=True,  return_state=True, name='lstm{}'.format(i)) \
            (x,   initial_state=[Add()([state_h , syllabus_dense[i]]), Add()([state_c , syllabus_dense[i]])])

        last_states_hs.append(x_state_h)
        last_states_cs.append(x_state_c)

    else:
        x, x_state_h, x_state_c = LSTM(lstm_dim , return_sequences=True,  return_state=True,name='lstm{}'.format(i)) \
            (x,   initial_state=[Add()([state_h , last_states_hs[i-1], syllabus_dense[i]]), Add()([state_c , last_states_cs[i-1], syllabus_dense[i]])])

        last_states_hs.append(x_state_h)
        last_states_cs.append(x_state_c)


    outputs.append(Dense(num_decoder_tokens[i], activation='softmax', name='predict{}'.format(i)) (x))

#model = Model([inputs, tuple(np.array(aux_inputs).tolist()), tuple(np.array(syllabus_inputs).tolist())], [tuple(np.array(outputs).tolist())], name='machine')
model = Model([inputs, aux_inputs[0],aux_inputs[1],aux_inputs[2], syllabus_inputs[0],syllabus_inputs[1],syllabus_inputs[2]], [outputs[0],outputs[1],outputs[2]], name='machine')

model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

params ={
            'num_encoder_tokens':num_encoder_tokens,
            'num_decoder_tokens':num_decoder_tokens,
            'max_encoder_seq_length':max_encoder_seq_length,
            'max_decoder_seq_length': max_decoder_seq_length,
            'input_token_index':input_token_index,
            'reverse_input_char_index':reverse_input_char_index,
            'target_token_index':target_token_index,
            'reverse_target_char_index':reverse_target_char_index,
            'number_of_output':HAIKU_LINES_NUM,
	    'embedding_matrix':embedding_matrix,
            #'latent_dim':latent_dim,
            #'num_syllabus':self.num_syllabus = len(syllabus)
        }

import r
DataGenerator = r.DataGenerator

training_generator = DataGenerator(X, y, params, batch_size=16 )
validation_generator = DataGenerator(X, y, params, batch_size=16)

print ('reached here?')
history = model.fit(training_generator, validation_data=validation_generator,  epochs=epochs, )
#history = model.fit_generator(training_generator, validation_data=validation_generator,  epochs=epochs, use_multiprocessing=True,)

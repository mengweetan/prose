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
Constant = tf.keras.initializers.Constant

EarlyStopping = tf.keras.callbacks.EarlyStopping
ModelCheckpoint = tf.keras.callbacks.ModelCheckpoint

pad_sequences = tf.keras.preprocessing.sequence.pad_sequences
Tokenizer = tf.keras.preprocessing.text.Tokenizer

from argparse import ArgumentParser

parser = ArgumentParser()

parser.add_argument("-o", "--out",dest="output",
                    help="location of model"
                    )

parser.add_argument("-b", "--build",
                    help="build embedded matrix"
                    )

parser.add_argument("-p", "--phrase", dest="phrase",
                help="text for seed"
                )

SEED_PHRASE = parser.parse_args().phrase

class Machine:
    def __init__(self):
        self.HAIKU_LINES_NUM = 3
        self.dataDir='data/'
        self.df = pd.read_csv(self.dataDir+'__INPUT.txt', sep = '\t')

        self.modelDir = parser.parse_args().output
        self.build_matrix = parser.parse_args().build if parser.parse_args().build  else None
        self._setup()

    @staticmethod
    def _process_target_texts(text):
        if 'start__ ' not in text:
            return 'start__ '+text+' __end'

    @staticmethod
    def _arrayOfInt(texts):
        try:
            array = texts.split(',')
            array = [int(i) for i in array]
        except:
            print (texts)
        return array

    def _setup(self):

        #self.df = self.df[:5000]
        print (self.df.info())

        t = Tokenizer()
        t.fit_on_texts([self.df['input_texts'][i] for i in range(self.df.shape[0])])
        vocab_size = len(t.word_index) + 1 # note - padded 1

        encoded_docs = t.texts_to_sequences([self.df['input_texts'][i] for i in range(self.df.shape[0])])
        max_len = max([len(l) for l in encoded_docs]) # get the max len
        padded_docs = pad_sequences(encoded_docs, maxlen=max_len, padding='post')

        if self.build_matrix:

            embeddings_index = dict()
            if os.name == 'nt': f = open(self.dataDir+'glove.6B/glove.6B.50d.txt',  encoding='utf-8') # try 50 dimension
            else: f = open(self.dataDir+'glove.6B/glove.6B.50d.txt')
            for line in f:
        	    values = line.split()
        	    word = values[0]
        	    coefs = np.asarray(values[1:], dtype='float32')
        	    embeddings_index[word] = coefs
            f.close()

            dimension_of_matrix =len(values)-1
            embedding_matrix = np.zeros((vocab_size, dimension_of_matrix)) # because we are using 50 dimension pre trained embedding
            for word, i in t.word_index.items():
                embedding_vector = embeddings_index.get(word)
                if embedding_vector is not None:
                    embedding_matrix[i] = embedding_vector

            np.savetxt(self.dataDir+'embedding_matrix.csv', embedding_matrix, delimiter=',')
            print ('done building embedding matrix')

        else: embedding_matrix = np.loadtxt(open(self.dataDir+'embedding_matrix.csv', "rb"), delimiter=",")



        self.df.line1=self.df['line1'].apply(self._process_target_texts)
        self.df.line2=self.df['line2'].apply(self._process_target_texts)
        self.df.line3=self.df['line3'].apply(self._process_target_texts)
        self.df.lib = self.df['lib'].apply(self._arrayOfInt)

        targetTexts = [self.df.line1, self.df.line2, self.df.line3 ]
        target_words = [ set(' '.join(item).split(' ')) for item in targetTexts ]

        num_encoder_tokens = vocab_size
        num_decoder_tokens = [len(i) for i in target_words ]

        for i in range(self.HAIKU_LINES_NUM):
            num_decoder_tokens[i] += 1

        target_token_index = [dict( [(char, i+1) for i, char in enumerate(j)]) for j in target_words]
        reverse_target_char_index = [dict((i, word) for word, i in j.items()) for j in target_token_index ]

        max_encoder_seq_length = max_len
        max_decoder_seq_length = [ max ([len(column[i].split(' ')) for i in range(self.df.shape[0])]) for column in targetTexts]

        input_token_index = t.word_index
        reverse_input_char_index = dict(map(reversed, t.word_index.items()))

        y = np.array( targetTexts)

        self.y = y.transpose()
        self.X = (padded_docs , self.df.lib)

        self.params ={
            'num_encoder_tokens':num_encoder_tokens,
            'num_decoder_tokens':num_decoder_tokens,
            'max_encoder_seq_length':max_encoder_seq_length,
            'max_decoder_seq_length': max_decoder_seq_length,
            'input_token_index':input_token_index,
            'reverse_input_char_index':reverse_input_char_index,
            'target_token_index':target_token_index,
            'reverse_target_char_index':reverse_target_char_index,
            'number_of_output':self.HAIKU_LINES_NUM,
            'HAIKU_LINES_NUM':self.HAIKU_LINES_NUM,
	    	'embedding_matrix':embedding_matrix,
                        #'latent_dim':latent_dim,
                        #'num_syllabus':self.num_syllabus = len(syllabus)
            }




    def train(self, epochs=5):



        latent_dim = self.params['embedding_matrix'].shape[1]
        dropout=0.1 #regularization , to prevent over fitting
        learning_rate = 0.0025

        import sys, os
        python_version = 2 if '2.' in sys.version.split('|')[0] else 3

        if python_version == 2: optimizer = 'rmsprop'
        else: optimizer =  tf.keras.optimizers.RMSprop(learning_rate=learning_rate, rho=0.9)

        # Model Architecture

        print (self.params['max_encoder_seq_length'])
        print (self.params['num_encoder_tokens'])


        inputs = Input(shape=(self.params['max_encoder_seq_length'],))

        x =  Embedding(self.params['num_encoder_tokens'], latent_dim,  embeddings_initializer=Constant(self.params['embedding_matrix']), input_length=self.params['max_encoder_seq_length'], trainable=False, mask_zero = True)(inputs)

        _ , state_h, state_c = LSTM(latent_dim,  return_state=True) (x)


        aux_inputs = [Input(shape=(None,), name='aux_input_{}'.format(i)) for i in range(self.params['HAIKU_LINES_NUM'])]

        syllabus_inputs = [Input(shape=(1,), name='syllabus_input_{}'.format(i)) for i in range(self.params['HAIKU_LINES_NUM'])]

        syllabus_dense = []
        last_states_hs = []
        last_states_cs = []
        outputs = []

        for i in range(self.params['HAIKU_LINES_NUM']):

            syllabus_dense.append( Dense(latent_dim , activation='softmax')  (syllabus_inputs[i]) )
            x = Embedding(self.params['num_decoder_tokens'][i]+1, latent_dim, mask_zero = True, name='line{}'.format(i)) (aux_inputs[i])
            if i == 0:
                x, x_state_h, x_state_c = LSTM(latent_dim , return_sequences=True,  return_state=True, name='lstm{}'.format(i)) \
                    (x,   initial_state=[Add()([state_h , syllabus_dense[i]]), Add()([state_c , syllabus_dense[i]])])

                last_states_hs.append(x_state_h)
                last_states_cs.append(x_state_c)

            else:
                x, x_state_h, x_state_c = LSTM(latent_dim , return_sequences=True,  return_state=True,name='lstm{}'.format(i)) \
                    (x,   initial_state=[Add()([state_h , last_states_hs[i-1], syllabus_dense[i]]), Add()([state_c , last_states_cs[i-1], syllabus_dense[i]])])

                last_states_hs.append(x_state_h)
                last_states_cs.append(x_state_c)


            outputs.append(Dense(self.params['num_decoder_tokens'][i], activation='softmax', name='predict{}'.format(i)) (x))

        #model = Model([inputs, tuple(np.array(aux_inputs).tolist()), tuple(np.array(syllabus_inputs).tolist())], [tuple(np.array(outputs).tolist())], name='machine')
        model = Model([inputs, aux_inputs[0],aux_inputs[1],aux_inputs[2], syllabus_inputs[0],syllabus_inputs[1],syllabus_inputs[2]], [outputs[0],outputs[1],outputs[2]], name='machine')

        model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
        model.summary()


        from r import DataGenerator
        #import r
        #DataGenerator = r.DataGenerator

        training_generator = DataGenerator(self.X, self.y, self.params, batch_size=64 )
        validation_generator = DataGenerator(self.X, self.y, self.params, batch_size=64)

        self.modelDir = self.modelDir if self.modelDir else 'model/haiku'
        if not os.path.exists(self.modelDir):
            os.makedirs(self.modelDir)

        mc = ModelCheckpoint(self.modelDir+'modelv2-best.h5')
        es = EarlyStopping(monitor='val_loss', mode='min', verbose=1)


        history = model.fit(training_generator, validation_data=validation_generator,  epochs=epochs, callbacks=[mc,es] )
        #history = model.fit_generator(training_generator, validation_data=validation_generator,  epochs=epochs, use_multiprocessing=True,)


        model.save(self.modelDir+'/modelv2-a.h5')
        print ('saved model in {}'.format(self.modelDir))
        #model.save_weights(modelDir+'modelv2_weights.h5')



        return history

if __name__ == "__main__":
    haiku = Machine()
    h = haiku.train(epochs=10)
    print (h.history)

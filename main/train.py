import tensorflow as tf
import numpy as np
import pandas as pd
import os, datetime

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

TensorBoard =  tf.keras.callbacks.TensorBoard

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

        self.df = self.df[:20000]
       
        print (self.df.info())

        t = Tokenizer()
        t.fit_on_texts([self.df['input_texts'][i] for i in range(self.df.shape[0])])
        
        #print (len(t.word_index) )
        t.word_index['__unknown'] = len(t.word_index) + 1 
        vocab_size = len(t.word_index)
        #print (vocab_size )
        
        vocab_size = len(t.word_index) + 1 # note - padded 1

        input_token_index = t.word_index
        reverse_input_char_index = dict(map(reversed, t.word_index.items()))

        
        encoded_docs = t.texts_to_sequences([self.df['input_texts'][i] for i in range(self.df.shape[0])])
        max_len = max([len(l) for l in encoded_docs]) # get the max len
        padded_docs = pad_sequences(encoded_docs, maxlen=max_len, padding='post')

        max_encoder_seq_length = max_len
        num_encoder_tokens = vocab_size

        embeddings_index = None
        dimension_of_matrix = None
        if self.build_matrix:

            embeddings_index = dict()
            if os.name == 'nt': f = open(self.dataDir+'glove.6B/glove.6B.50d.txt',  encoding='utf-8') # try 50 dimension
            else: f = open(self.dataDir+'glove.6B/glove.6B.50d.txt')
            for line in f:
        	    matrix_values = line.split()
        	    word = matrix_values[0]
        	    coefs = np.asarray(matrix_values[1:], dtype='float32')
        	    embeddings_index[word] = coefs
            f.close()

            dimension_of_matrix =len(matrix_values)-1
            embedding_matrix = np.zeros((vocab_size, dimension_of_matrix)) # because we are using 50 dimension pre trained embedding
            for word, i in t.word_index.items():
                embedding_vector = embeddings_index.get(word)
                if embedding_vector is not None:
                    embedding_matrix[i] = embedding_vector

            np.savetxt(self.dataDir+'{}_embedding_matrix.csv'.format(vocab_size), embedding_matrix, delimiter=',')



            print ('done building embedding matrix')

        else: embedding_matrix = np.loadtxt(open(self.dataDir+'{}_embedding_matrix.csv'.format(vocab_size), "rb"), delimiter=",")

        # now do the decoding matrix stuff

        self.df.line1=self.df['line1'].apply(self._process_target_texts)
        self.df.line2=self.df['line2'].apply(self._process_target_texts)
        self.df.line3=self.df['line3'].apply(self._process_target_texts)
        self.df.lib = self.df['lib'].apply(self._arrayOfInt)

        targetTexts = [self.df.line1, self.df.line2, self.df.line3 ]
        #target_words = [ set(' '.join(item).split(' ')) for item in targetTexts ]
        #target_words = [ set(' '.join(item).split(' ')).union(set(['__unknown'])) for item in targetTexts ] # add unknown token to vocab of target texts

        
        # num_decoder_tokens = [len(i) for i in target_words ]

        # target_token_index = [dict( [(char, i+1) for i, char in enumerate(j)]) for j in target_words]
        #reverse_target_char_index = [dict((i, word) for word, i in j.items()) for j in target_token_index ]

        #for i in range(self.HAIKU_LINES_NUM):
        #     num_decoder_tokens[i] += 1

        target_embedding_matrix = []
        
        #for i in range(self.HAIKU_LINES_NUM):

            



       
        max_decoder_seq_length = [ max ([len(column[i].split(' ')) for i in range(self.df.shape[0])]) for column in targetTexts]

        num_decoder_tokens, target_token_index , reverse_target_char_index, decode_tokenizer, decoded_docs, decoded_padded_docs  = [], [], [], [], [], []

        for i in range(self.HAIKU_LINES_NUM):
            decode_tokenizer.append(Tokenizer())
            decode_tokenizer[i].fit_on_texts([targetTexts[i][j] for j in range(self.df.shape[0])])

            

            decode_tokenizer[i].word_index['__unknown'] = len(decode_tokenizer[i].word_index) + 1 
            decode_tokenizer[i].word_index['start__'] = len(decode_tokenizer[i].word_index) + 1 
            decode_tokenizer[i].word_index['__end'] = len(decode_tokenizer[i].word_index) + 1 

            num_decoder_tokens.append(len(decode_tokenizer[i].word_index))
       
            num_decoder_tokens[i] += 1 # note - padded 1

            target_token_index.append(decode_tokenizer[i].word_index)
            reverse_target_char_index.append( dict(map(reversed, decode_tokenizer[i].word_index.items())))

            decoded_docs.append( decode_tokenizer[i].texts_to_sequences([ targetTexts[i][j] for j in range(self.df.shape[0])])  )
            decoded_padded_docs.append( pad_sequences(decoded_docs[i], maxlen=max_decoder_seq_length[i], padding='post'))

            if self.build_matrix:
                target_embedding_matrix.append(np.zeros((num_decoder_tokens[i], dimension_of_matrix)) )# because we are using 50 dimension pre trained embedding
                for word, j in target_token_index[i].items():
                    embedding_vector = embeddings_index.get(word)
                    if embedding_vector is not None:
                        target_embedding_matrix[i][j] = embedding_vector

                np.savetxt(self.dataDir+'{}_{}_target_embedding_matrix.csv'.format(num_decoder_tokens[i],i), target_embedding_matrix[i], delimiter=',')
                print ('finished target matrix')


            else: target_embedding_matrix.append ( np.loadtxt(open(self.dataDir+'{}_{}_target_embedding_matrix.csv'.format(num_decoder_tokens[i],i), "rb"), delimiter=",") )

        targetTexts =   np.array(decoded_docs) 
        #print (targetTexts.shape)
        #print (decoded_docs[0])
        #print (decoded_docs[0].shape)
        #print (decoded_padded_docs.shape)
        #print (decoded_padded_docs[2].shape)
        targetTexts = np.array([i.tolist() for i in decoded_padded_docs])
        
        #print (targetTexts.shape)
        #print (decoded_docs[0])
        #print (decoded_padded_docs[0].shape)
        #targetTexts =   np.array(decoded_padded_docs)
        #print (targetTexts.shape)

        #print (target_token_index[0])
       
        




        

        '''
        y = np.array( targetTexts)
        print ('y', y.shape)

        self.y = y.transpose()
        '''

        self.y = targetTexts.transpose()
        print (self.y.shape)
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
            'target_embedding_matrix':target_embedding_matrix,

                        #'latent_dim':latent_dim,
                        #'num_syllabus':self.num_syllabus = len(syllabus)
            }




    def train(self, epochs=5):



        latent_dim = self.params['embedding_matrix'].shape[1]
        dropout=0.1 #regularization , to prevent over fitting
        learning_rate = 0.005

        import sys, os
        python_version = 2 if '2.' in sys.version.split('|')[0] else 3

        if python_version == 2: optimizer = 'rmsprop'
        else: optimizer =  tf.keras.optimizers.RMSprop(learning_rate=learning_rate, rho=0.9)
        #else: optimizer =  tf.keras.optimizers.Adagrad()
	    
		

        # Model Architecture

        print ('max_encoder_seq_length', self.params['max_encoder_seq_length'])
        print ('num_encoder_tokens', self.params['num_encoder_tokens'])


        inputs = Input(shape=(self.params['max_encoder_seq_length'],))

        x =  Embedding(self.params['num_encoder_tokens'], latent_dim,  embeddings_initializer=Constant(self.params['embedding_matrix']), input_length=self.params['max_encoder_seq_length'], trainable=False, mask_zero = True)(inputs)

        _ , state_h, state_c = LSTM(latent_dim,  return_state=True) (x)


        aux_inputs = [Input(shape=(self.params['max_decoder_seq_length'][i],), name='aux_input_{}'.format(i)) for i in range(self.params['HAIKU_LINES_NUM'])]

        syllabus_inputs = [Input(shape=(1,), name='syllabus_input_{}'.format(i)) for i in range(self.params['HAIKU_LINES_NUM'])]

        syllabus_dense = []
        last_states_hs = []
        last_states_cs = []
        outputs = []
        xs = []

        for i in range(self.params['HAIKU_LINES_NUM']):

            syllabus_dense.append( Dense(latent_dim , activation='softmax')  (syllabus_inputs[i]) )
            #xs.append(Embedding(self.params['num_decoder_tokens'][i]+1, latent_dim, embeddings_initializer=Constant(self.params['target_embedding_matrix'][i]), input_length=self.params['max_decoder_seq_length'][i], trainable=False, \
            #    mask_zero = True, name='line{}'.format(i)) (aux_inputs[i]) )
            xs.append(Embedding(self.params['num_decoder_tokens'][i], latent_dim, embeddings_initializer=Constant(self.params['target_embedding_matrix'][i]), input_length=self.params['max_decoder_seq_length'][i], trainable=False, \
                mask_zero = True, name='line{}'.format(i)) (aux_inputs[i]) )
            if i == 0:
                x, x_state_h, x_state_c = LSTM(latent_dim , return_sequences=True,  return_state=True, name='lstm{}'.format(i)) \
                    (xs[i],   initial_state=[Add()([state_h , syllabus_dense[i]]), Add()([state_c , syllabus_dense[i]])])

                last_states_hs.append(x_state_h)
                last_states_cs.append(x_state_c)

            else:
                x, x_state_h, x_state_c = LSTM(latent_dim , return_sequences=True,  return_state=True,name='lstm{}'.format(i)) \
                    (xs[i],   initial_state=[Add()([last_states_hs[i-1], syllabus_dense[i]]), Add()([last_states_cs[i-1], syllabus_dense[i]])])
                    #(x,   initial_state=[Add()([state_h , last_states_hs[i-1], syllabus_dense[i]]), Add()([state_c , last_states_cs[i-1], syllabus_dense[i]])])
                    

                last_states_hs.append(x_state_h)
                last_states_cs.append(x_state_c)


            outputs.append(Dense(self.params['num_decoder_tokens'][i], activation='softmax', name='predict{}'.format(i)) (x))

        #model = Model([inputs, tuple(np.array(aux_inputs).tolist()), tuple(np.array(syllabus_inputs).tolist())], [tuple(np.array(outputs).tolist())], name='machine')
        model = Model([inputs, aux_inputs[0],aux_inputs[1],aux_inputs[2], syllabus_inputs[0],syllabus_inputs[1],syllabus_inputs[2]], [outputs[0],outputs[1],outputs[2]], name='machine')

        #model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
        model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
        model.summary()


        from r import DataGenerator
        #import r
        #DataGenerator = r.DataGenerator

        training_generator = DataGenerator(self.X, self.y, self.params, batch_size=128 )
        validation_generator = DataGenerator(self.X, self.y, self.params, batch_size=128)

        self.modelDir = self.modelDir if self.modelDir else 'model/haiku'
        if not os.path.exists(self.modelDir):
            os.makedirs(self.modelDir)

        mc = ModelCheckpoint(self.modelDir+'/modelv2-best.h5',monitor='val_loss', verbose=1, save_best_only=True)
        es = EarlyStopping(monitor='predict0_loss', mode='min', verbose=1)

        log_dir = "data\\logs\\fit\\" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        #log_dir = self.dataDir+"logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        tfc = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
        # launch at console: tensorboard --logdir data/logs/fit
        
        history = model.fit(training_generator, validation_data=validation_generator,  shuffle=True, epochs=epochs, callbacks=[mc,es,tfc] )
        
        ##history = model.fit_generator(training_generator, validation_data=validation_generator,  epochs=epochs, use_multiprocessing=True,)


        model.save(self.modelDir+'/modelv2-b.h5')
        print ('saved model in {}'.format(self.modelDir))
       



        return history

    def plot(self,h) :

        import matplotlib.pyplot as plt
        print (h.history)
        

        f, axs = plt.subplots(2,figsize=(15,15))

        legend =[]
        for i in range(3):
            axs[0].plot(h.history['predict{}_loss'.format(i)],'blue')
            axs[0].plot(h.history['val_predict{}_loss'.format(i)],'black')
            legend.append('train_{}_loss'.format(i))
            legend.append('val_{}_loss'.format(i))
            axs[1].plot(h.history['predict{}_accuracy'.format(i)],'red')

        #plt.figure()
        #
        #ticks =  ax.get_xticks()
        #ax.set_xticklabels([int(abs(tick)) for tick in ticks])
        axs[0].legend(legend, loc='upper right')
        #axs[0].ylabel('Accuracy')
        #axs[0].xlabel('Epoch')

        

        '''
        legend =[]
        for i in range(3):
            plt.plot(h.history['predict{}_loss'.format(i)],'blue')
            plt.plot(h.history['val_predict{}_loss'.format(i)],'black')
            legend.append('train_{}_loss'.format(i))
            legend.append('val_{}_loss'.format(i))

        #plt.figure()
        #
        #ticks =  ax.get_xticks()
        #ax.set_xticklabels([int(abs(tick)) for tick in ticks])
        plt.legend(legend, loc='upper right')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        '''
        plt.savefig(self.modelDir+'/HISTORY.png')
        

if __name__ == "__main__":
    haiku = Machine()
    h = haiku.train(epochs=10)
    haiku.plot(h)
    

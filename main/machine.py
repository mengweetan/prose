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

class Machine:
    def __init__(self, size=None, build_matrix=False):
        
        self.HAIKU_LINES_NUM = 3
        file_path = os.getcwd()+'/ml/'
        self.dataDir=file_path +'prose/data/'
        self.df = pd.read_csv(self.dataDir+'__INPUT.txt', sep = '\t')
        self.size = size if size else self.df.shape[0]
        # self.modelDir = parser.parse_args().output
        #self.build_matrix = parser.parse_args().build if parser.parse_args().build  else None
        self.build_matrix = build_matrix # no need for now
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

        if self.size: self.df = self.df[:self.size]
       
        print ('based on datafame {}'.format(self.df.shape))

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

        targetTexts = np.array([i.tolist() for i in decoded_padded_docs])
        
       
        #print (self.df.lib.shape)
        self.y = targetTexts.transpose()
        #print (self.y.shape)
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

            }

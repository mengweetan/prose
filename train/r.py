import pandas as pd
import numpy as np
import string


import tensorflow as tf




Sequence = tf.keras.utils.Sequence

class DataGenerator(Sequence):
    def __init__(self, X, y, params,   to_fit=True, batch_size=32,  shuffle=True):

        self.X = X
        self.y =y
        self.params = params
        self.to_fit = to_fit
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.on_epoch_end()



    def __getitem__(self, index):

        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        inputs, aux_inputs, outputs, syllabus_inputs = self._generate(self.X, self.y, self.params )

        #return [0,0,0,0,0,0,0],[0,0,0]
        return [inputs, aux_inputs[0],aux_inputs[1],aux_inputs[2], syllabus_inputs[0],syllabus_inputs[1],syllabus_inputs[2]], [outputs[0],outputs[1],outputs[2]]
        #return [inputs, np.array(np.array(aux_inputs).tolist()), tuple(syllabus_inputs)], [np.array(np.array(outputs).tolist())]
        #return [inputs, tuple(np.array(aux_inputs).tolist())],[tuple(np.array(outputs).tolist())]

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.X[0]))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def _generate(self, X, y, params):
        batch_size = self.batch_size



        for j in range(0, len(X[0]), batch_size):
            _input = np.zeros((batch_size, params['max_encoder_seq_length']),dtype='float32')

            # old model
            _aux_input = [np.zeros((batch_size, params['max_decoder_seq_length'][ii]),dtype='float32') for ii in range(params['number_of_output'])]
            # new model


            _syllabus_inputs = [np.zeros((batch_size,),dtype='float32') for ii in range(params['number_of_output'])]

            _output = [np.zeros((batch_size, params['max_decoder_seq_length'][ii], params['num_decoder_tokens'][ii]),dtype='float32')  for ii in range(params['number_of_output'])]

            for i, (input_text, target_text) in enumerate(zip(X[0][j:j+batch_size], y[j:j+batch_size])):

                #for t, word in enumerate(input_text.split(' ')):
                for t, word in enumerate(input_text):
                    
                    #_input[i, t] = params['input_token_index'][word]  # encoder input seq
                    _input[i, t] = params['embedding_matrix'][word] # maybe??

                #_input[i, t+1:] = params['input_token_index']['']
                for z, line_text in enumerate(target_text):

                    # old model
                    for t, word in enumerate(line_text.split(' ')): # because we are using multiple outputs
                        # decoder_target_data is ahead of decoder_input_data by one timestep
                        _aux_input[z][i, t] = params['target_token_index'][z][word] # decoder input seq
                        if t>0:
                            # decoder_target_data will be ahead by one timestep
                            # and will not include the start character.
                            _output[z][i, t - 1, params['target_token_index'][z][word]] = 1.

                    #_aux_input[z][i, t+1:] = params['target_token_index'][z][''] #
                    #_output[z][i, t:, params['target_token_index'][z]['']] = 1.




            return _input, _aux_input, _output, _syllabus_inputs


    def __len__(self):
        return int(np.floor(len(self.X[0]) / self.batch_size))


class lineMaker:
    def __init__(self, encoder_model, decoder_model, params ):

        self.encoder_model = encoder_model
        self.decoder_model = decoder_model
        self.params = params



    def imagine(self, seed, haiku_style=[5,7,5]):
        from .utils import syllable_count

        def _sample(preds, temperature=0.3):

            preds = np.asarray(preds).astype('float64')
            preds = np.log(preds) / temperature # 1 is temperature
            exp_preds = np.exp(preds)
            preds = exp_preds / np.sum(exp_preds)
            probas = np.random.multinomial(1, preds, 1)
            return np.argmax(probas)




        print (seed)
        serialised = pd.Series([seed])
        haiku_style = np.array(haiku_style)


        X =np.array([serialised,haiku_style])

        #_haiku_style=[5,7,5]

        d = DataGenerator(X, '', self.params,  batch_size=1)

        [s,_,_,_, s1,s2,s3], _ = d.__getitem__(0)



        state_values =  self.encoder_model.predict([s, np.array([haiku_style[0]]), np.array([haiku_style[1]]), np.array([haiku_style[2]])]) # stndard 5-7-5 haiku!


        ts = np.zeros((1,1))
        ts[0, 0] = self.params['target_token_index'][0]['start__']
        ts1 = np.zeros((1,1))
        ts1[0,0] = (self.params['target_token_index'][1]['start__'])
        ts2 = np.zeros((1,1))
        ts2[0,0] = (self.params['target_token_index'][2]['start__'])
        '''

        ts = []
        for z in range(3):
            ts.append(self.params['target_token_index'][z]['start__'])
        '''

        syllabus_count = [0,0,0]
        #_haiku_0,  _haiku_1,  _haiku_2 = '','',''
        stop_condition = False
        try_count = 0

        _haiku = {}
        max_syllabus = {}

        for i in range(3):
           _haiku[i] = []
           max_syllabus[i] = False


        while not stop_condition:
            r = ([ts,ts1,ts2] + state_values + [np.array([haiku_style[0]]), np.array([haiku_style[1]])  , np.array([haiku_style[2]])])
            #r = ([ts[0],ts[1],ts[2]] + state_values + [np.array([haiku_style[0]]), np.array([haiku_style[1]])  , np.array([haiku_style[2]])])
            print ('ok...')
            i, ii, iii, h ,c = self.decoder_model.predict(r)
            for j in range(3):
                sampled_token_index = np.zeros((1,))

                if j == 0:

                    #print (self.params['reverse_target_char_index'][j][_index])
                    sampled_token_index = _sample(i[0, -1, :])

                    #sampled_token_index = np.argmax(i[0, -1, :])
                elif j == 1:
                    #sampled_token_index  = np.argmax(ii[0, -1, :])
                    sampled_token_index = _sample(ii[0, -1, :])
                elif j == 2:
                    #sampled_token_index  = np.argmax(iii[0, -1, :])
                    sampled_token_index = _sample(iii[0, -1, :])

                word = (self.params['reverse_target_char_index'][j][sampled_token_index]).split('/')[0]

                if not max_syllabus[j] and word not in _haiku[j]:
                    _haiku[j].append(word)
                    syllabus_count[j] += syllable_count(word)

                syllabus_limit = 7 if j==1 else 5
                if syllabus_count[j] >= syllabus_limit or word == '__end': max_syllabus[j] = True
                #if word == '__end': max_syllabus[j] = True
                else:

                    if j == 0:
                        ts[0, 0] = np.zeros((1,1))
                        ts[0, 0] = sampled_token_index
                    elif j == 1:
                        ts1[0, 0] = np.zeros((1,1))
                        ts1[0, 0] = sampled_token_index
                    elif j == 2:
                        ts2[0, 0] = np.zeros((1,1))
                        ts2[0, 0] = sampled_token_index
                    '''
                    ts[j] = sampled_token_index
                    '''

            if ( max_syllabus[0] ==   max_syllabus[1] == max_syllabus[2] == True) \
                or (sum([len(''.join(_haiku[i])) for  i in range(3)]) > 140) :

                '''
                if  syllabus_count[0] != 5 or syllabus_count[1] != 7 or syllabus_count[2] != 5:


                    print ( ' '.join( [ _haiku_0 , _haiku_1, _haiku_2]) )

                    max_syllabus = {}
                    max_syllabus[0],max_syllabus[1],max_syllabus[2] = False, False, False
                    syllabus_count = [0,0,0]
                    _haiku_0,  _haiku_1,  _haiku_2 = '','',''
                    try_count +=1
                    print ('try again, after {}'.format(try_count))
                    if try_count > 10:
                        print ('give up!')
                        stop_condition = True
                        break

                else:
                '''
                stop_condition = True
                break

            state_values = [h, c]

        return '\n'.join( [ ' '.join(_haiku[i]).replace('__end','') for i in range(3)])

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf

Model = tf.keras.models.Model
Input = tf.keras.layers.Input
LSTM = tf.keras.layers.LSTM
Embedding = tf.keras.layers.Embedding
Dense = tf.keras.layers.Dense
concatenate = tf.keras.layers.concatenate
Flatten = tf.keras.layers.Flatten
Add =  tf.keras.layers.Add
Constant = tf.keras.initializers.Constant

pad_sequences = tf.keras.preprocessing.sequence.pad_sequences
Tokenizer = tf.keras.preprocessing.text.Tokenizer

import numpy as np





from  train import Machine
haiku = Machine()
embedding_matrix = haiku.params['embedding_matrix']
latent_dim = embedding_matrix.shape[1]
HAIKU_LINES_NUM = haiku.params['HAIKU_LINES_NUM']

def getInferenceModels():


    '''
    num_encoder_tokens = 50000
    num_decoder_tokens = 50000

    inputs = Input(shape=(None,))
    aux_inputs = [Input(shape=(haiku.params['max_encoder_seq_length'],), name='haikuLine_{}'.format(i)) for i in range(HAIKU_LINES_NUM)]
    syllabus_inputs = [Input(shape=(1,),  dtype='int32', name='syllabus_input_{}'.format(i)) for i in range(HAIKU_LINES_NUM)]

    x =  Embedding(num_encoder_tokens+1, latent_dim, mask_zero = True)(inputs)
    _ , state_h, state_c = LSTM(latent_dim,  return_state=True) (x)

    encoder_model = Model([inputs, syllabus_inputs[0],syllabus_inputs[1],syllabus_inputs[2]], [state_h, state_c])

    '''
    load_model = tf.keras.models.load_model
    model = load_model('model/haiku/modelv2-b.h5',custom_objects={'tf': tf}, compile=False)
    encoder_model = Model( [ model.inputs[0], model.inputs[4], model.inputs[5], model.inputs[6]], [model.layers[4].output[1],model.layers[4].output[2] ])



    #decoder model

    #inputs = Input(shape=(None,))
    '''
    inputs = Input(shape=(haiku.params['max_encoder_seq_length'], ), name='input_before_embed')
    
    x =  Embedding(haiku.params['num_encoder_tokens'], latent_dim, mask_zero = True)(inputs)
    _ , state_h, state_c = LSTM(latent_dim,  return_state=True) (x)
    '''

    #aux_inputs = [Input(shape=(haiku.params['max_encoder_seq_length'],), name='haikuLine_{}'.format(i)) for i in range(HAIKU_LINES_NUM)]
    aux_inputs = [Input(shape=(1,), name='proseLine_{}'.format(i)) for i in range(HAIKU_LINES_NUM)]
    syllabus_inputs = [Input(shape=(1,),  dtype='int32', name='syllabus_input_{}'.format(i)) for i in range(HAIKU_LINES_NUM)]


    decoder_state_input_h = Input(shape=(latent_dim,))
    decoder_state_input_c = Input(shape=(latent_dim,))
    decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]

    syllabus_state_hs = []
    syllabus_state_cs = []
    syllabus_dense = []
    xs = []

    last_states_hs = []
    last_states_cs = []

    decoder_outputs2 = []


    for i in range(HAIKU_LINES_NUM):

        
        x2 = Embedding(haiku.params['num_decoder_tokens'][i]+1, latent_dim,  name='syllabus{}'.format(i)) (syllabus_inputs[i]) 
        #x2 = Embedding(haiku.params['num_decoder_tokens'][i]+1, latent_dim, mask_zero = True, name='line{}'.format(i)) (syllabus_inputs[i]) 
        x2, s_state_h, s_state_c =  LSTM(latent_dim, return_sequences=True, return_state=True, name='lstm_post_line_emb{}'.format(i)) (x2)
        syllabus_state_hs.append(s_state_h)
        syllabus_state_cs.append(s_state_c)

        syllabus_dense.append( Dense(latent_dim , activation='softmax')  (syllabus_inputs[i]) )

        #x = Embedding(haiku.params['num_decoder_tokens'][i]+1, latent_dim, mask_zero = True, name='line{}'.format(i)) (aux_inputs[i])
        xs.append(Embedding(haiku.params['num_decoder_tokens'][i], latent_dim, embeddings_initializer=Constant(haiku.params['target_embedding_matrix'][i]), input_length=haiku.params['max_decoder_seq_length'][i], trainable=False, \
                mask_zero = True, name='line{}'.format(i)) (aux_inputs[i]) )

        if i == 0:
            x, x_state_h, x_state_c = LSTM(latent_dim , return_sequences=True,  return_state=True, name='lstm{}'.format(i)) \
                (xs[i],   initial_state=[Add()([decoder_state_input_h , syllabus_dense[i]]), Add()([decoder_state_input_c , syllabus_dense[i]])])

            last_states_hs.append(x_state_h)
            last_states_cs.append(x_state_c)

        else:
            x, x_state_h, x_state_c = LSTM(latent_dim , return_sequences=True,  return_state=True,name='lstm{}'.format(i)) \
                (xs[i],   initial_state=[Add()([last_states_hs[i-1], syllabus_dense[i]]), Add()([last_states_cs[i-1], syllabus_dense[i]])])
                #(x,   initial_state=[Add()([decoder_state_input_h , last_states_hs[i-1], syllabus_dense[i]]), Add()([decoder_state_input_c , last_states_cs[i-1], syllabus_dense[i]])])
                

            last_states_hs.append(x_state_h)
            last_states_cs.append(x_state_c)

        Dropout = tf.keras.layers.Dropout
        x = Dropout(0.2, input_shape=(haiku.params['num_decoder_tokens'][i],)) (x)
        decoder_outputs2.append(Dense(haiku.params['num_decoder_tokens'][i], activation='sigmoid', name='predict{}'.format(i)) (x))
        # x = Dense(30, activation='relu', name='predictd{}'.format(i)) (x)
        # decoder_outputs2.append(Dense(1, activation='sigmoid', name='predict{}'.format(i)) (x))

        #decoder_outputs2.append(Dense(haiku.params['num_decoder_tokens'][i], activation='softmax', name='predict{}'.format(i)) (x))


    decoder_model = Model(
        [aux_inputs] + decoder_states_inputs + [syllabus_inputs],
        #[decoder_outputs2[0] , decoder_outputs2[1] , decoder_outputs2[2]] + [state_h2, state_c2])
        [tuple(np.array(decoder_outputs2).tolist())] + [x_state_h, x_state_c]) # the last identity of the last line
    #decoder_model.load_weights(self.data_dir+'model8_weights.h5')
    decoder_model.summary()
   

    return encoder_model, decoder_model



    def work_on(seed):
        from r import lineMaker

        encoder_model, decoder_model = getInferenceModels()
        lm = lineMaker( encoder_model, decoder_model, haiku.params )
        result = lm.imagine(SEED_PHRASE)

        # API key: 5OEsg3kEBkhZE7LE
        # Password: Well done!

        return (result)

if __name__ == "__main__":

    from train import SEED_PHRASE 
    print (SEED_PHRASE)


    from r import lineMaker
    #seed ="long john silver"
 


    encoder_model, decoder_model = getInferenceModels()
    lm = lineMaker( encoder_model, decoder_model, haiku.params )
    result = lm.imagine(SEED_PHRASE)

    print (result)

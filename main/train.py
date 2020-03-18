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

modelDir = parser.parse_args().output

file_path = os.getcwd()+'/ml/'

modelDir = modelDir if modelDir else file_path+'prose/model/haiku'
if not os.path.exists(modelDir):
    os.makedirs(modelDir)

SEED_PHRASE = parser.parse_args().phrase
build_matrix = True if parser.parse_args().build  else False


def train(machine, epochs=5):



    latent_dim = machine.params['embedding_matrix'].shape[1]
    #dropout=0.1 #regularization , to prevent over fitting
    learning_rate = 0.0025

    import sys, os
    python_version = 2 if '2.' in sys.version.split('|')[0] else 3

    #if python_version == 2: optimizer = 'rmsprop'
    #else: optimizer =  tf.keras.optimizers.RMSprop(learning_rate=learning_rate, rho=0.9)
    #else: optimizer =  tf.keras.optimizers.Adagrad()
    
    
    optimizer =  tf.keras.optimizers.RMSprop(learning_rate=learning_rate, rho=0.9)
    optimizer= 'adam'

    # Model Architecture

    print ('max_encoder_seq_length', machine.params['max_encoder_seq_length'])
    print ('num_encoder_tokens', machine.params['num_encoder_tokens'])


    inputs = Input(shape=(machine.params['max_encoder_seq_length'],))

    x =  Embedding(machine.params['num_encoder_tokens'], latent_dim,  embeddings_initializer=Constant(machine.params['embedding_matrix']), input_length=machine.params['max_encoder_seq_length'], trainable=False, mask_zero = True)(inputs)

    _ , state_h, state_c = LSTM(latent_dim,  return_state=True) (x)


    aux_inputs = [Input(shape=(machine.params['max_decoder_seq_length'][i],), name='aux_input_{}'.format(i)) for i in range(machine.params['HAIKU_LINES_NUM'])]

    syllabus_inputs = [Input(shape=(1,), name='syllabus_input_{}'.format(i)) for i in range(machine.params['HAIKU_LINES_NUM'])]

    syllabus_dense = []
    last_states_hs = []
    last_states_cs = []
    outputs = []
    xs = []

    for i in range(machine.params['HAIKU_LINES_NUM']):

        syllabus_dense.append( Dense(latent_dim , activation='softmax')  (syllabus_inputs[i]) )
        #xs.append(Embedding(machine.params['num_decoder_tokens'][i]+1, latent_dim, embeddings_initializer=Constant(machine.params['target_embedding_matrix'][i]), input_length=machine.params['max_decoder_seq_length'][i], trainable=False, \
        #    mask_zero = True, name='line{}'.format(i)) (aux_inputs[i]) )
        xs.append(Embedding(machine.params['num_decoder_tokens'][i], latent_dim, embeddings_initializer=Constant(machine.params['target_embedding_matrix'][i]), input_length=machine.params['max_decoder_seq_length'][i], trainable=False, \
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


        # add drop out??
        Dropout = tf.keras.layers.Dropout
        x = Dropout(0.2, input_shape=(machine.params['num_decoder_tokens'][i],)) (x)
        #x = Dense(30, activation='relu', name='predictd{}'.format(i)) (x)
        outputs.append(Dense(machine.params['num_decoder_tokens'][i], activation='sigmoid', name='predict{}'.format(i)) (x))

        # before experiment
        #outputs.append(Dense(machine.params['num_decoder_tokens'][i], activation='softmax', name='predict{}'.format(i)) (x))

    #model = Model([inputs, tuple(np.array(aux_inputs).tolist()), tuple(np.array(syllabus_inputs).tolist())], [tuple(np.array(outputs).tolist())], name='machine')
    model = Model([inputs, aux_inputs[0],aux_inputs[1],aux_inputs[2], syllabus_inputs[0],syllabus_inputs[1],syllabus_inputs[2]], [outputs[0],outputs[1],outputs[2]], name='machine')

    


    from r import DataGenerator
    #import r
    #DataGenerator = r.DataGenerator

    training_generator = DataGenerator(machine.X, machine.y, machine.params, batch_size=16 )
    validation_generator = DataGenerator(machine.X, machine.y, machine.params, batch_size=16)





    # tensorboard stuff
    from tensorboard.plugins.hparams import api as hp
    HP_DROPOUT = hp.HParam('dropout', hp.RealInterval(0.1, 0.2))
    HP_OPTIMIZER = hp.HParam('optimizer', hp.Discrete(['adam', 'sgd','rmsprop']))
    METRIC_ACCURACY = 'accuracy'

    with tf.summary.create_file_writer('logs/hparam_tuning').as_default():
        hp.hparams_config(
            hparams=[HP_DROPOUT, HP_OPTIMIZER],
            metrics=[hp.Metric(METRIC_ACCURACY, display_name='Accuracy')],
        )


    #logdir = "data\\logs\\fit\\" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    logdir = machine.dataDir+"logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    file_writer = tf.summary.create_file_writer(logdir + "/metrics")
    file_writer.set_as_default()
    def lr_schedule(epochs):
        """
        Returns a custom learning rate that decreases as epochs progress.
        """
        learning_rate = 0.05
        if epochs > 10:
            learning_rate = 0.02
        if epochs > 20:
            learning_rate = 0.01
        if epochs > 50:
            learning_rate = 0.005

        tf.summary.scalar('learning rate', data=learning_rate, step=epochs)
        return learning_rate

    lrc = tf.keras.callbacks.LearningRateScheduler(lr_schedule)


    tfc = tf.keras.callbacks.TensorBoard(log_dir=logdir, histogram_freq=1)
    # launch at console: tensorboard --logdir data/logs/fit
    mc = ModelCheckpoint(modelDir+'/modelv2-best.h5',monitor='val_loss', verbose=1, save_best_only=True)
    es = EarlyStopping(monitor='loss', mode='min', verbose=1, patience=5)



    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    # model.summary()

    history = model.fit(training_generator, validation_data=validation_generator, shuffle=True, epochs=epochs, callbacks=[mc,es,lrc] )
    
    #history = model.fit_generator(training_generator, validation_data=validation_generator,  epochs=epochs, use_multiprocessing=True, callbacks=[mc,es,tfc,lrc])


    model.save(modelDir+'/{}_modelv2.h5'.format(machine.size))
    print ('saved model in {}'.format(modelDir))
    



    return history

def plotG(h) :

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
    plt.savefig(modelDir+'/HISTORY.png')
        

if __name__ == "__main__":
    from machine import Machine
    haiku = Machine(build_matrix=build_matrix, size=10000 )
    
    #h = haiku.train(epochs=20)
    h = train(haiku, epochs= 1)
    plotG(h)
    

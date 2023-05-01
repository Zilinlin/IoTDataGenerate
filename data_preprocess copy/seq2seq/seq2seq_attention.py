import sys
import logging

import numpy as np
from tensorflow.keras.optimizers import Adam
import keras
from keras.models import Sequential, Model, load_model
from keras.layers import LSTM, Dense, RepeatVector, TimeDistributed, Input, BatchNormalization, Activation, dot, concatenate
from keras.callbacks import EarlyStopping
from keras import metrics
import tensorflow as tf
from sklearn.utils import shuffle
from matplotlib import pyplot as plt
from math import floor, ceil
import os

N_HIDDEN=128
EPOCH=30
THRESHOLD=0.01

import numpy as np
import copy

class Analyzer:
    def __init__(self, name):
        self.name = name
        self.model = None

    def get_name(self):
        return self.name

    def model_exists(self):
        ret = False
        if self.model:
            ret = True
        return ret

    def learning(self, sequence, config):
        pass

    def analysis(self, sequence, config):
        pass

    def print_infection_information(self, results, config):
        ofname = "{}/{}".format(config["home"], config["output"])
        print ("===== Infection Information =====")
        cnt = 0
        checked = []
        windows = []

        for prob, window in results:
            serial = window.get_serial_number()
            if serial not in checked:
                checked.append(serial)
                windows.append((prob, window))

        with open(ofname, "w") as of:
            of.write("Number, Confidence, Flow, Start Time, End Time, Answer Label, Classified Label, Classified Probability\n")
            for prob, infection in windows:
                cnt += 1

                print ("{}> Serial Number: {}".format(cnt, infection.get_serial_number()))
                print ("  - Confidence: {}".format(prob))
                print ("  - Flow: {}".format(infection.get_flow_info()))
                print ("  - Start Time: {}".format(infection.get_start_time()))
                print ("  - End Time: {}".format(infection.get_end_time()))
                print ("  - Answer Label: {}".format(infection.get_label("infection")))
                print ("  - Classified Label: {}".format(infection.get_labeled("infection")))
                print ("  - Classified Probability: {}".format(infection.get_probability("infection")))
                of.write("{}, {}, {}, {}, {}, {}, {}, {}\n".format(cnt, prob, infection.get_flow_info(), infection.get_start_time(), infection.get_end_time(), infection.get_label("infection"), infection.get_labeled("infection"), infection.get_probability("infection")))


class Seq2seqAttention(Analyzer):
    def __init__(self, name):
        super().__init__(name)        

    def truncate(self, x, y, idxs_order, slen=100):
        in_, out_, truncated_idxs = [], [], []

        for i in range(len(x) - slen + 1):
            in_.append(x[i:i+slen])
            out_.append(y[i:(i+slen)])
            truncated_idxs.append(idxs_order[i:(i+slen)])
        return np.array(in_), np.array(out_), np.array(truncated_idxs)


    def replace_sublist_inplace(self, A, B, replace_idxs):
        if (len(A)==len(B)):
            for i, idx in enumerate(replace_idxs):
                #print(idx)
                #print(A)
                #print(B)
                A[i] = B[idx]
        else:
            raise(Exception("Arrays A and B should have same length"))

    def permute_sublist_inplace(self, A, permute_idxs):
        A_orig = copy.copy(A)
        self.replace_sublist_inplace(A, A_orig, permute_idxs)


    def permute_truncated(self, X_in, X_out, truncated_idxs, slen=10, inplace=False):
        enable_permute_prints = False
        if not inplace:
            X_in = copy.copy(X_in)
            truncated_idxs = copy.copy(truncated_idxs)
        for x_seq_in, x_seq_out, seq_idxs in zip(X_in, X_out, truncated_idxs):
            repeating_seq = []
            permute_idxs = []
            i = 0
            current_label = x_seq_out[i]
            #label_next = current_label
            repeating_seq.append(i)
            i+=1
            while i < slen:
                prev_label = current_label
                current_label = x_seq_out[i]
                if i < 20 and enable_permute_prints:
                    #assert(0)
                    print(i, current_label, prev_label)

                if prev_label != current_label: 
                    if i < 20 and enable_permute_prints:
                        print(repeating_seq)
                    permute_idxs = permute_idxs + list(np.random.permutation(repeating_seq))
                    #x_seq_in[repeating_seq] = x_seq_in[idx_permutation]
                    repeating_seq = []
                    repeating_seq.append(i)
                    i+=1
                else:
                    repeating_seq.append(i)
                    i+=1 
                if i < 20 and enable_permute_prints:
                    print(repeating_seq)
                
            permute_idxs = permute_idxs + list(np.random.permutation(repeating_seq))
            if i < 20 and enable_permute_prints:
                print("permuting {} with idxs {}".format(x_seq_in, permute_idxs))
                print("permuting {} with idxs {}".format(seq_idxs, permute_idxs))
            self.permute_sublist_inplace(x_seq_in, permute_idxs)    
            self.permute_sublist_inplace(seq_idxs, permute_idxs)
            #print(seq_idxs)
        if not inplace:
            return X_in, truncated_idxs

    def probability_based_embedding(self, p, d):
        ret = 0
        pr = {}

        tmp = zip(range(4), p)
        order = [k for k, _ in sorted(tmp, key=lambda x: x[1], reverse=True)]
                    
        ru = 0
        rd = 0
        for i in range(4):
            r = round(p[i], d)
            c = ceil(p[i] * 10 ** d) / 10 ** d
            f = floor(p[i] * 10 ** d) / 10 ** d
            if r - c == 0:
                ru += 1
            elif r - f == 0:
                rd += 1

        lst = []
        if ru >= 2:
            for i in range(4):
                if i == order[-1]:
                    lst.append(floor(p[i] * 10 ** d) / 10 ** d)
                else:
                    lst.append(round(p[i], d))
            if sum(lst) > 0.999 and sum(lst) < 1.001:
                for i in range(4):
                    pr[i] = lst[i]
            else:
                for i in range(4):
                    if i == order[-1] or i == order[-2]:
                        pr[i] = floor(p[i] * 10 ** d) / 10 ** d
                    else:
                        pr[i] = round(p[i], d)

        elif rd >= 2:
            for i in range(4):
                if i == order[0]:
                    lst.append(ceil(p[i] * 10 ** d) / 10 ** d)
                else:
                    lst.append(round(p[i], d))
            if sum(lst) > 0.999 and sum(lst) < 1.001:
                for i in range(4):
                    pr[i] = lst[i]
            else:
                for i in range(4):
                    if i == order[0] or i == order[1]:
                        pr[i] = ceil(p[i] * 10 ** d) / 10 ** d
                    else:
                        pr[i] = round(p[i], d)

        for i in [2, 1, 0, 3]:
            ret *= 10 ** d
            ret += round(pr[i] * (10 ** d), 0)
        
        return ret
    
    # Please implement the following functions
    def learning(self, windows, labels, config, permute_truncated=True):
        logging.debug('Learning: {}'.format(self.get_name()))
        #states = sequence.get_sequence()
        #states = windows
        #features = len(states[0].get_code())
        slen = config["sequence_length"]
        rv = config["rv"]
        permute_truncated = config["permute_truncated"]
        print("Permute truncated is {}".format(permute_truncated))

        in_, out_ = [], []
        idx_order = []
        idx = 0
        use_prob_embedding = config["use_prob_embedding"]
        print("Use prob embedding is {}".format(use_prob_embedding))

        for idx, (prob, label) in enumerate(zip(windows, labels)):
            #prob = state.get_probability_vector()
            p = prob
            if use_prob_embedding:
                #print("4 dim event embedded = {}".format(prob))
                #print("state/window {}".format(window))
                p = self.probability_based_embedding(prob, rv)
                #if idx<20:
                #    print("4 dim event embedded = {}".format(p))
            #label = state.get_label("infection")
            logging.info("{}> {} : {}".format(idx, p, label))
            in_.append(p)
            out_.append([label])
            idx_order.append(idx)
        '''d
        print("in is {}".format(in_[:100]))
        print("in shape is {}".format(np.array(in_).shape))
        print("out is {}\n".format([x for x in out_[:100]]))
        print("out shape is {} \n".format(np.array(out_).shape))
        print("slen is {}".format(slen))
        '''

        X_in, X_out, truncated_idxs = self.truncate(in_, out_, idx_order, slen=slen)
        X_out_labels = np.array(X_out)[:,:,0].tolist()

        if permute_truncated:
            print("Permute Truncated is enabled")
            X_in, perm_truncated_idxs = self.permute_truncated(X_in, X_out_labels, truncated_idxs, slen=slen, inplace=False)
        else:
            print("Permute Truncated is disabled")

        if use_prob_embedding:
            X_in = np.expand_dims(X_in, axis=-1)

        #print("truncated idxs are {}".format(truncated_idxs[:20]))
        #if permute_truncated:
            #print("perm truncated idxs are {}".format(perm_truncated_idxs[:20]))
        #print("x_in is {}".format(X_in[:20]))
        #print("x_in shape is {}".format(np.array(X_in).shape))
        #print("x_out is {}\n".format([x for x in X_out[:20]]))
        #print("x_out shape is {} \n".format(np.array(X_out).shape))


        input_train = Input(shape=(X_in.shape[-2], X_in.shape[-1]))
        #print('input train is', input_train)
        output_train = Input(shape=(X_out.shape[-2], X_out.shape[-1]))

        encoder_stack_h, encoder_last_h, encoder_last_c = LSTM(
                N_HIDDEN, activation='relu', dropout=0.2, recurrent_dropout=0.2,
                return_sequences=True, return_state=True)(input_train)

        encoder_last_h = BatchNormalization(momentum=0.6)(encoder_last_h)
        encoder_last_c = BatchNormalization(momentum=0.6)(encoder_last_c)

        decoder_input = RepeatVector(output_train.shape[1])(encoder_last_h)
        decoder_stack_h = LSTM(N_HIDDEN, activation='relu', dropout=0.2, recurrent_dropout=0.2,
                return_state=False, return_sequences=True)(decoder_input, initial_state=[encoder_last_h, encoder_last_c])
        
        attention = dot([decoder_stack_h, encoder_stack_h], axes=[2,2])
        attention = Activation('softmax')(attention)

        context = dot([attention, encoder_stack_h], axes=[2,1])
        context = BatchNormalization(momentum=0.6)(context)

        decoder_combined_context = concatenate([context, decoder_stack_h])
        out = TimeDistributed(Dense(output_train.shape[2]))(decoder_combined_context)
        out = Activation('sigmoid')(out)


        self.model = Model(inputs=input_train, outputs=out)
        opt = Adam(learning_rate=0.01, clipnorm=1)
        binary_crossentropy = tf.keras.losses.BinaryCrossentropy(
                                from_logits=False,
                                label_smoothing=0.0,
                                #axis=-1,
                                reduction="auto",
                                name="binary_crossentropy",
                            )

        metrics = ['accuracy', tf.keras.metrics.AUC(), tf.keras.metrics.Recall(), tf.keras.metrics.Precision()]
        #metrics = ['accuracy']
        self.model.compile(loss=binary_crossentropy, optimizer=opt, metrics=metrics)
        #self.model.compile(loss='mean_squared_error', optimizer=opt, metrics=metrics)
        self.model.summary()

        #es = EarlyStopping(monitor='val_loss', mode='min', patience=50)
        X_in_before_fit = X_in
        X_in_before_fit, X_out = shuffle(X_in_before_fit,X_out)
        '''for i in X_in_before_fit[:20]:
            #print(i)
            for j in i:
                j = np.array(j)'''

        #print(X_in_before_fit[:30])
        '''for i in X_in:
            for j in i:
                for k in j:
                    k=list(k)'''
                    #print(k)
        #for i in X_in:
        #    X_in_before_fit.append(i[:,0])
        #X_in_before_fit = np.array(X_in[:, :, 0])
        #np.random.shuffle(X_in_before_fit
        #history = self.model.fit(X_in_before_fit, X_out[:, :, :1], validation_split=0.5, epochs=EPOCH, verbose=1, callbacks=[es], batch_size=100)
        history = self.model.fit(X_in_before_fit, X_out[:, :, :1], validation_split=0.2, epochs=EPOCH, verbose=1, batch_size=100)
        #dir = 'emb{}_perm{}'.format(use_prob_embedding, permute_truncated)
        #os.makedirs(dir, exist_ok=True)
        #for metric in ['loss', 'accuracy', 'auc', 'recall', 'precision']:
        #    self.plot_keras_metric(metric, history, dir+'/'+metric+'.png')

    def plot_keras_metric(self, metric_name, history, file_name):

        plt.plot(history.history[metric_name])
        plt.plot(history.history['val_'+metric_name])
        plt.title('model '+ metric_name)
        plt.ylabel(metric_name)
        plt.xlabel('epoch')
        plt.legend(['train', 'val'], loc='upper left')
        plt.savefig(file_name)
        plt.show()  
        plt.clf()


    def analysis(self, windows, labels, config):
        '''
        gets states from test sequence
        gets per-step-detectors prob for each state
        performes inference on each state 
        '''

        '''
        gets states from test sequence        
        '''
        logging.debug('Analysis: {}'.format(self.get_name()))
        #states = sequence.get_sequence()
        #states = windows
        #features = len(states[0].get_code())
        slen = config["sequence_length"]
        rv = config["rv"]
        permute_truncated = config["permute_truncated"]
        print("Permute truncated is {}".format(permute_truncated))

        in_, out_ = [], []
        idx_order = []
        idx = 0
        use_prob_embedding = config["use_prob_embedding"]
        print("Use prob embedding is {}".format(use_prob_embedding))

        for idx, (prob, label) in enumerate(zip(windows, labels)):
            #prob = state.get_probability_vector()
            p = prob
            if use_prob_embedding:
                print("4 dim event embedded = {}".format(prob))
               # print("state/window {}".format(window))
                p = self.probability_based_embedding(prob, rv)
                if idx<20:
                    print("4 dim event embedded = {}".format(p))
            #label = state.get_label("infection")
            logging.info("{}> {} : {}".format(idx, p, label))
            in_.append(p)
            out_.append([label])
            idx_order.append(idx)

        '''
        performs inference on X_in, formed from in_ sequences truncated
        '''
        X_in, _, _ = self.truncate(in_, in_, idx_order, slen=slen)
        y_pred = self.model.predict(X_in)



        '''acumulates predictions'''
        idx = 0
        prediction_1 = []
        prediction_0 = []
        predictions = {}
        
        for pred in y_pred:

            '''iterates through remaining truncated seq len if surpassing the limit'''
            if len(y_pred) - idx > slen:
                lst = range(len(pred))
            else:
                lst = range(len(y_pred)-idx)

            '''acumulates truncated predictions: e.g. {idx1: [1,1,0,0], idx2: [1,0,0,1], idx3: [0,0,1,1], ...]'''
            for i in lst:
                if idx + i not in predictions:
                    predictions[idx + i] = []
                predictions[idx + i].append(pred[i][0])
            idx += 1
            
        '''looks like it takes the average of predictions for each truncated sequence? not sure'''
        results = []
        for idx in range(len(windows) - slen + 1):
            res = sum(predictions[idx])/len(predictions[idx])
            results.append(res)
        ret_probs = []
        ret_idxs = []
        for idx in range(len(results)):
            #states[idx].set_hidden_label_int(0)
            if results[idx] > THRESHOLD:
                #states[idx].set_hidden_label_int(1)
                prob = results[idx]
                ret_probs.append(prob)
                ret_idxs.append(idx)

        #self.print_infection_information(ret, config)
        return ret_probs, ret_idxs
# -*- encoding:utf-8 -*-
__author__ = 'Suncong Zheng'

import pickle
import numpy as np
from PrecessEEdata import get_data_e2e
from Evaluate import evaluavtion_triple
from keras.models import Sequential
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
from keras.layers.core import Dropout
from decodelayer import ReverseLayer2, LSTMDecoder_tag, Position_Embedding, Attention
from keras.layers import Bidirectional, TimeDistributed, Dense, Activation
from keras_contrib.layers.crf import CRF
from keras import optimizers


def get_training_batch_xy_bias(inputsX, inputsY, max_s, max_t, batchsize, vocabsize, target_idex_word, lossnum,
                               shuffle=False):
    assert len(inputsX) == len(inputsY)
    indices = np.arange(len(inputsX))
    if shuffle:
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputsX) - batchsize + 1, batchsize):
        excerpt = indices[start_idx:start_idx + batchsize]
        x = np.zeros((batchsize, max_s)).astype('int32')
        y = np.zeros((batchsize, max_t, vocabsize + 1)).astype('int32')
        for idx, s in enumerate(excerpt):
            x[idx, ] = inputsX[s]
            for idx2, word in enumerate(inputsY[s]):
                targetvec = np.zeros(vocabsize + 1)
                wordstr = ''
                if word != 0:
                    wordstr = target_idex_word[word]
                if wordstr.__contains__("E"):
                    targetvec[word] = lossnum
                else:
                    targetvec[word] = 1
                y[idx, idx2, ] = targetvec
        yield x, y


def save_model(nn_model, NN_MODEL_PATH):
    nn_model.save_weights(NN_MODEL_PATH, overwrite=True)


def creat_binary_tag_LSTM(sourcevocabsize, targetvocabsize, source_W, input_seq_lenth, output_seq_lenth,
    hidden_dim, emd_dim, loss='categorical_crossentropy', optimizer='rmsprop'):

    # # input_dim是词汇表大小，output_dim是词向量维度，input_length是输入序列长度
    l_A_embedding = Embedding(input_dim=sourcevocabsize + 1,
                              output_dim=emd_dim,
                              input_length=input_seq_lenth,
                              mask_zero=True,
                              weights=[source_W])

    model = Sequential()
    model.add(l_A_embedding)
    model.add(Position_Embedding(hidden_dim, 'concat'))
    model.add(Dropout(0.3))

    # Random embedding
    model.add(Bidirectional(LSTM(hidden_dim, return_sequences=True, dropout=0.2)))
    model.add(Attention(2, 300))

    model.add(TimeDistributed(Dense(targetvocabsize + 1)))
    # crf = CRF(targetvocabsize + 1, sparse_target=False)
    # model.add(crf)
    model.add(Activation('softmax'))
    model.summary()
    model.compile(loss=loss, optimizer=optimizer)
    # model.compile(loss=crf.loss_function, optimizer=optimizers.RMSprop(lr=0.0001, rho=0.9, epsilon=1e-06),
    #               metrics=[crf.accuracy])
    return model


def test_model(nn_model, testdata, index2word, resultfile=''):
    index2word[0] = ''
    testx = np.asarray(testdata[0], dtype="int32")
    testy = np.asarray(testdata[1], dtype="int32")

    batch_size = 50
    testlen = len(testx)
    testlinecount = 0
    if len(testx) % batch_size == 0:
        testnum = len(testx)/batch_size
    else:
        extra_test_num = batch_size - len(testx) % batch_size
        extra_data = testx[:extra_test_num]
        testx = np.append(testx, extra_data, axis=0)
        extra_data = testy[:extra_test_num]
        testy = np.append(testy, extra_data, axis=0)

        testnum = len(testx) / batch_size

    testresult = []
    for n in range(0, int(testnum)):
        xbatch = testx[n*batch_size:(n+1)*batch_size]
        ybatch = testy[n*batch_size:(n+1)*batch_size]
        predictions = nn_model.predict(xbatch)

        for si in range(0, len(predictions)):
            if testlinecount < testlen:
                sent = predictions[si]
                ptag = []
                for word in sent:
                    next_index = np.argmax(word)
                    if next_index != 0:
                        next_token = index2word[next_index]
                        ptag.append(next_token)
                senty = ybatch[si]
                ttag = []
                for word in senty:
                    next_token = index2word[word]
                    ttag.append(next_token)
                result = []
                result.append(ptag)
                result.append(ttag)
                testlinecount += 1
                testresult.append(result)
    pickle.dump(testresult, open(resultfile, 'wb'))
    P, R, F = evaluavtion_triple(testresult)
    return P, R, F


def train_e2e_model(eelstmfile, modelfile, resultdir, npochos,
                    lossnum=1, batch_size=256, retrain=False):

    # load training data and test data
    traindata, testdata, source_W, source_vob, sourc_idex_word, target_vob, target_idex_word, max_s, k \
        = pickle.load(open(eelstmfile, 'rb'))

    # train model
    x_train = np.asarray(traindata[0], dtype="int32")
    y_train = np.asarray(traindata[1], dtype="int32")

    nn_model = creat_binary_tag_LSTM(sourcevocabsize=len(source_vob), targetvocabsize=len(target_vob),
                                     source_W=source_W, input_seq_lenth=max_s, output_seq_lenth=max_s,
                                     hidden_dim=k, emd_dim=k)
    if retrain:
        nn_model.load_weights(modelfile)

    epoch = 0
    save_inter = 2
    saveepoch = save_inter
    maxF = 0
    history_sum = []
    while(epoch < npochos):
        epoch = epoch + 1
        for x, y in get_training_batch_xy_bias(x_train, y_train, max_s, max_s, batch_size, len(target_vob),
                                               target_idex_word, lossnum, shuffle=True):
            history = nn_model.fit(x, y, batch_size=batch_size, epochs=1, verbose=1)
        history_sum.append(history.history['loss'])
        if epoch == saveepoch:
            resultfile = resultdir + "result3-" + str(saveepoch)
            saveepoch += save_inter
            P, R, F = test_model(history.model, testdata, target_idex_word, resultfile)
            if F > maxF:
                maxF = F
                save_model(history.model, modelfile)
            print('Epoch: [{}/{}], P R F {:.4f} {:.4f} {:.4f}'.format(
                epoch, npochos, P, R, F))
    print(history_sum)
    return nn_model


def infer_e2e_model(eelstmfile, lstm_modelfile, resultfile):

    traindata, testdata, source_W, source_vob, sourc_idex_word, target_vob, target_idex_word, max_s, k \
        = pickle.load(open(eelstmfile, 'rb'))
    nnmodel = creat_binary_tag_LSTM(sourcevocabsize=len(source_vob), targetvocabsize=len(target_vob),
                                    source_W=source_W, input_seq_lenth=max_s, output_seq_lenth=max_s,
                                    hidden_dim=k, emd_dim=k)

    nnmodel.load_weights(lstm_modelfile)
    P, R, F = test_model(nnmodel, testdata, target_idex_word, resultfile)
    print(P, R, F)


if __name__ == "__main__":
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = "3"

    alpha = 1
    maxlen = 337
    trainfile = "./data/demo/train_tag-all.json"
    testfile = "./data/demo/test_tag.json"
    w2v_file = "./data/demo/w2v.pkl"
    e2edatafile = "./data/demo/result3/e2edata.pkl"
    modelfile = "./data/demo/result3/e2e_lstmb_model.pkl"
    resultdir = "./data/demo/result3/"

    retrain = False
    valid = False
    if not os.path.exists(e2edatafile):
        print("Precess lstm data....")
        get_data_e2e(trainfile, testfile, w2v_file, e2edatafile, maxlen=maxlen)
    if not os.path.exists(modelfile):
        print("Lstm data has extisted: "+e2edatafile)
        print("Training EE model....")
        train_e2e_model(e2edatafile, modelfile, resultdir, npochos=100, lossnum=alpha, retrain=False)
    else:
        if retrain:
            print("ReTraining EE model....")
            train_e2e_model(e2edatafile, modelfile, resultdir, npochos=100, lossnum=alpha, retrain=retrain)

    # retrain = True
    # valid = True
    # infer_e2e_model(e2edatafile, modelfile, resultdir+'result-last')

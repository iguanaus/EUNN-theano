import theano, cPickle
import theano.tensor as T
import numpy as np
from fftconv import cufft, cuifft
import keras.backend as K



def compute_cost_t(lin_output,y_t):
    RNN_output = T.nnet.softmax(lin_output)
    CE = T.nnet.categorical_crossentropy(RNN_output, y_t)
    cost_t = CE.mean()
    acc_t =(T.eq(T.argmax(RNN_output, axis=-1), y_t)).mean(dtype=theano.config.floatX)
    return cost_t, acc_t

def permute_list12(s):
    #s = 8
    ind1 = range(s)
    ind2 = range(s)

    for i in range(s):

        if i%2 == 1:
            ind1[i] = ind1[i] - 1
            if i == s -1:
                continue
            else:
                ind2[i] = ind2[i] + 1
        else:
            ind1[i] = ind1[i] + 1
            if i == 0: 
                continue
            else:
                ind2[i] = ind2[i] - 1
    #print("Index permutation",[ind1,ind2])
    return [ind1, ind2]

def permute_approx(s):
    def ind_s(k):
        if k==0:
            return np.array([[1,0]])
        else:
            temp = np.array(range(2**k))
            list0 = [np.append(temp + 2**k, temp)]
            list1 = ind_s(k-1)
            for i in range(k):
                list0.append(np.append(list1[i],list1[i] + 2**k))
            return list0

    t = ind_s(int(np.log2(s/2)))

    ind_list5 = []
    for i in range(int(np.log2(s))):
        ind_list5.append(t[i])

    ind_list6 = []
    for i in range(int(np.log2(s))):
        ind = np.array([])
        for j in range(2**i):
            ind = np.append(ind, np.array(range(0, s, 2**i)) + j).astype(np.int32)

        ind_list6.append(ind)
    return ind_list5, ind_list6


def permute_list34(s):

    ind3 = []
    ind4 = []

    for i in range(s/2):
        ind3.append(i)
        ind3.append(i + s/2)

    ind4.append(0)
    for i in range(s/2 - 1):
        ind4.append(i + 1)
        ind4.append(i + s/2)
    ind4.append(s - 1)

    return [ind3, ind4]
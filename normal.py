import random
random.seed(1)

import numpy as np

def get_random_arr(l):
    return [random.uniform(-0.99, 0.99) for x in xrange(l)]

def p(arr):
    for _ in arr:
        print _


def sig(x,deriv=False):
    if(deriv==True):
        return x*(1-x)

    return 1/(1+np.exp(-x))

DIM = 3
TRAIN_NUM = 4
# training set
X = [
     [0,0,1],
     [0,1,1],
     [1,0,1],
     [1,1,1]
     ]

# result for training set
Y = [ 0, 1, 1, 0 ]

# init random synapse weight
syn0 = [
        [0.8, 0.8, 0.8],
        [0.8, 0.8, 0.8],
        [0.8, 0.8, 0.8],
        [0.8, 0.8, 0.8]
    ]
syn1 = [
        0.8,
        0.8,
        0.8,
        0.8
    ]


syn0 = [ get_random_arr(3) for x in xrange(TRAIN_NUM)]
syn1 = get_random_arr(TRAIN_NUM)


for j in xrange(1000000):
    l0 = X

    # init with 0s
    l1 = [[ 0 for a in xrange(TRAIN_NUM)] for x in xrange(TRAIN_NUM)]

    # calculate l1
    for i in xrange(TRAIN_NUM):
        for k in xrange(TRAIN_NUM):
            l1[i][k] = sig(sum([a * b for a, b in zip(X[i], syn0[k])]))
    
    # calculate l2
    l2 = [0 for a in xrange(TRAIN_NUM)]
    for i in xrange(TRAIN_NUM):
        l2[i] = sig(sum([a * b for a,b in zip(l1[i], syn1)]))    

    # calculate l2_error and delta
    l2_error = [ Y[i] - l2[i] for i in xrange(TRAIN_NUM) ]

    if (j% 1000) == 0:
        # print "Error:" + str(np.mean(np.abs(l2_error)))
        print "Error: ", sum([ abs(a)/4 for a in l2_error]) 

    l2_delta = [ l2_error[i] * sig(l2[i],deriv=True) for i in xrange(TRAIN_NUM) ]


    l1_error = [ 
        [syn1[i] * l2_delta[j] for i in xrange(TRAIN_NUM)] 
        for j in xrange(TRAIN_NUM)
        ]
    l1_delta = [[ 0 for a in xrange(TRAIN_NUM)] for x in xrange(TRAIN_NUM)]

    for i in xrange(TRAIN_NUM):
        for j in xrange(TRAIN_NUM):
            l1_delta[i][j] = l1_error[i][j] * sig(l1[i][j], deriv=True)


    syn1_delta = [0 for x in xrange(TRAIN_NUM)]
    for i in xrange(TRAIN_NUM):
        l1_col = [ x[i] for x in l1 ]
        syn1_delta[i] = sum([a * b for a,b in zip(l1_col, l2_delta)])

    syn0_delta = [ [0 for i in xrange(TRAIN_NUM)] for j in xrange(DIM)]
    l0T = [ [l0[j][i] for j in xrange(TRAIN_NUM)] for i in xrange(DIM)]

    for i in xrange(DIM):
        for j in xrange(TRAIN_NUM):
            ithrow = l0T[i]
            jthcol = [l1_delta[k][j] for k in xrange(TRAIN_NUM)] 
            syn0_delta[i][j] = sum([a * b for a, b in zip(ithrow, jthcol)])

    for i in xrange(4):
        for j in xrange(3):
            syn0[i][j] += syn0_delta[j][i]

    for i in xrange(4):
        syn1[i] += syn1_delta[i]


    # p(syn0)
    # print syn1
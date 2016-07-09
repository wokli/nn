import numpy as np
# sigmoid function
def sig(x,deriv=False):
    if(deriv==True):
        return x*(1-x)
    return 1/(1+np.exp(-x))


import random
random.seed(1)

TRAIN_NUM = 4
XLEN = 3

# training set
X = [ [0, 0, 1],
      [0, 1, 1],
      [1, 0, 1],
      [1, 1, 1] ]

# result for training set
Y = [ 0, 0, 1, 1 ]

# init random synapse weight
syn = [ 0.1 for x in xrange(XLEN)]

for iter in xrange(10000):

    # first layer
    l0 = X
    # second layer
    l1 = [0 for x in xrange(TRAIN_NUM)]

    for i in xrange(TRAIN_NUM):
        l1[i] = sig(sum(
                a * b for a, b in zip(X[i], syn)
            ))
    # print l1


    l1_error = [ a - b for a,b in zip(Y, l1)]
    # print l1_error

    l1_delta = [ a * b for a,b in zip(
        l1_error, 
        [sig(a, True) for a in l1]
        )]
    # print l1_delta

    weight_upd = [0, 0, 0]
    for i in xrange(XLEN):
        weight_upd[i] = sum( a * b for a, b in zip(
                [x[i] for x in X],
                l1_delta
            ))

    # print weight_upd

    for i in xrange(XLEN):
        syn[i]+=weight_upd[i]

print 'syn:', syn
print 'Output after training:'
print l1
    

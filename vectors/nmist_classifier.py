import pickle, gzip, os
from urllib import request
from pylab import imshow, show, cm
import numpy as np

f = gzip.open('mnist.pkl.gz', 'rb')
train_set, valid_set, test_set = pickle.load(f, encoding ='latin1')
f.close()


def get_image (number):
    (X, y) = [img[number] for img in train_set ]
    return (np.array(X), y)

def view_image ( number ):
    (X, y) = get_image ( number )
    print (" Label : %s" % y)
    imshow (X.reshape(28 ,28), cmap=cm.gray)
    show()

i = get_image(0)
view_image(i)
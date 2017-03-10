import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

data_f = open("./tensorflow_DATA/X/X_Test_case_01.dat")
read_data = np.loadtxt(data_f)
data_f.close()

np.random.seed(1)

def f(x, a, b):
    n = train_X.size
    vals = np.zeros((1, n))

    for i in range(0, n):
        ax = np.multiply(a, x.item(i))
        val = np.add(ax, b)
        vals[0, i] = val
    return vals

Wref = 0.7
bref = -1.
n = 20
noise_var = 0.001
train_X = np.random.random((1, 20))
ref_Y   = f(train_X, Wref, bref)
train_Y = ref_Y + np.sqrt(noise_var)*np.random.randn(1, n)
n_samples = train_X.size

print("")
print (" Type of 'train_X' is ", train_X)
print (" Shape of 'train_X' is %s" % (train_X.shape,))
print (" Type of 'train_Y' is ", type(train_Y))
print (" Shape of 'train_Y' is %s" % (train_Y.shape,))

plt.plot(train_X, 'o', label='Input Data')

pylab.show()
#plt.savefig('myfig')
#plt.legend()
#plt.show()

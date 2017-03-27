import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

data_f = open("./tensorflow_DATA/relative/relTest_case_01.dat")
read_data = np.loadtxt(data_f)
data_f.close()

lon_def = 125.079590
lat_def = 36.578830

num_steps = 5
batch_size = 1
rnn_size = 5

y_lon = read_data[:,0]
y_lat = read_data[:,1]
velo_U = read_data[:,2]
velo_V = read_data[:,3]
wind_U = read_data[:,4]
wind_V = read_data[:,5]

W1 = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
W2 = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
W3 = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
W4 = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
b = tf.Variable(tf.random_uniform([1], -1.0, 1.0))

hypothesis = W1*velo_U + W2*velo_V + W3*wind_U + W4*wind_V + b

cost_lon = tf.reduce_mean(tf.square(hypothesis - y_lon))
cost_lat = tf.reduce_mean(tf.square(hypothesis - y_lat))

rate = tf.Variable(0.01)
optimizer = tf.train.GradientDescentOptimizer(rate)
train_lon = optimizer.minimize(cost_lon)
train_lat = optimizer.minimize(cost_lat)

#rnn_cell = tf.nn.rnn_cell.BasicRNNCell(rnn_size)
#state = tf.zeros([batch_size, rnn_cell.state_size])

#loss = tf.nn.seq2seq.sequence_loss_by_example(y_lon, hypothesis)

#cost = tf.reduce_sum(loss) / batch_size
#train_op = tf.train.RMSPropOptimizer(0.01, 0.9).minimize(cost)

init = tf.initialize_all_variables()

sess = tf.Session()
sess.run(init)

for step in range(4000) :
    sess.run(train_lon)
    sess.run(train_lat)

    if step%200 == 0 :
        print(step, "Lon : ", sess.run(cost_lon), "Lat : ", sess.run(cost_lat))

#plt.plot(y_lon, y_lat, 'o', label='Input Data')
#plt.legend()
#plt.show()

sess.close()

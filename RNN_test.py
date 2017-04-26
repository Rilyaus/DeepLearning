import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
#import xlwt

data_trainSet = open("./tensorflow_DATA/cross_4attr/relTrain_case06.dat")
data_testSet = open("./tensorflow_DATA/cross_4attr/relTest_case06.dat")
read_data_trainSet = np.loadtxt(data_trainSet)
read_data_testSet = np.loadtxt(data_testSet)

data_trainSet.close()
data_testSet.close()

lon_def = 125.079590
lat_def = 36.578830

num_steps = 5
batch_size = 1
rnn_size = 4
learing_rate = 0.01
epoch = 1100

lon_train = read_data_trainSet[:,0]
lat_train = read_data_trainSet[:,1]
lon_test = read_data_testSet[:,0]
lat_test = read_data_testSet[:,1]
y_train = read_data_trainSet[:,0:2]
y_test = read_data_testSet[:,0:2]

lon_train = lon_train.reshape(len(np.atleast_1d(lon_train)), 1)
lat_train = lat_train.reshape(len(np.atleast_1d(lat_train)), 1)

y_train = y_train.reshape(len(np.atleast_1d(y_train)), 2, 1)
x_train = read_data_trainSet[:,2:6]
x_train = x_train.reshape(len(np.atleast_1d(x_train)), 4, 1)

lon_test = lon_test.reshape(len(np.atleast_1d(lon_test)), 1)
lat_test = lat_test.reshape(len(np.atleast_1d(lat_test)), 1)

y_test = y_test.reshape(len(np.atleast_1d(y_test)), 2, 1)
x_test = read_data_testSet[:,2:6]
x_test = x_test.reshape(len(np.atleast_1d(x_test)), 4, 1)


# ---------- RNN Network ---------- #

# input place holders
data = tf.placeholder(tf.float32, [None, 4, 1])
y_lon = tf.placeholder(tf.float32, [None, 1])
y_lat = tf.placeholder(tf.float32, [None, 1])
y = tf.placeholder(tf.float32, [None, 2, 1])

# RNN Network
rnn_cell = tf.contrib.rnn.BasicLSTMCell(6)
val, _state = tf.nn.dynamic_rnn(rnn_cell, data, dtype=tf.float32)
pred = tf.contrib.layers.fully_connected(val[:, -1], 2, activation_fn=None)

# cost
#cost_lon = tf.reduce_mean(tf.square(pred[0] - y_lon))
#cost_lat = tf.reduce_mean(tf.square(pred[1] - y_lat))
cost = tf.reduce_mean(tf.add(tf.square(pred[0] - y[0]), tf.square(pred[1] - y[1])))

# optimizer
optimizer = tf.train.AdamOptimizer(learing_rate)
#train_lon = optimizer.minimize(cost_lon)
#train_lat = optimizer.minimize(cost_lat)
train = optimizer.minimize(cost)

# RMSE / MAE
targets = tf.placeholder(tf.float32, [None, 1])
predictions = tf.placeholder(tf.float32, [None, 1])
rmse = tf.sqrt(tf.reduce_mean(tf.square(targets - predictions)))
mae = tf.reduce_mean(tf.abs(targets - predictions))


with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)

# Training
    for step in range(2000) :
        _, step_loss = sess.run([train, cost], feed_dict={data:x_train, y:y_train})

        if step%500 == 0 and step != 0 :
            print step,", cost_lon :", step_loss
# Test
    test_pred = sess.run(pred, feed_dict={data:x_test})
    np.savetxt("file1.csv", test_pred, delimiter=",")

#print 'Epoch :', epoch

"""
with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)

# Training
    for step in range(2000) :
        _, step_loss_lon = sess.run([train_lon, cost_lon], feed_dict={data:x_train, y_lon:lon_train})

        if step%500 == 0 and step != 0 :
            print step,", cost_lon :", step_loss_lon
# Test
    test_pred = sess.run(pred, feed_dict={data:x_test})
    np.savetxt("file1.csv", test_pred, delimiter=",")
    #mae_lon = sess.run(mae, feed_dict={targets:lon_test, predictions: test_pred})
    #print "MAE_lon :", mae_lon


with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)

    for step in range(2000) :
        _, step_loss_lat = sess.run([train_lat, cost_lat], feed_dict={data:x_train, y_lat:lat_train})

        if step%500 == 0 and step != 0 :
            print step,", cost_lat :", step_loss_lat
# Test
    test_pred = sess.run(pred, feed_dict={data:x_test})
    np.savetxt("file2.csv", test_pred, delimiter=",")
    #mae_lat = sess.run(mae, feed_dict={targets:lat_test, predictions: test_pred})
    #print "MAE_lat :", mae_lat

"""

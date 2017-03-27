import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

data_f = open("./tensorflow_DATA/X/X_Test_case_01.dat")
read_data = np.loadtxt(data_f)
data_f.close()

x_data = [4.0391, 1.3197, 9.5613, 0.5978, 3.5316, 0.1540, 1.6899, 7.3172, 4.5092, 2.9632]
y_data = [11.4215, 10.0112, 30.2991, 1.0625, 13.1776, -3.1976, 6.7367, 23.8550, 14.8951, 11.6137]


W = tf.Variable(tf.random_uniform([1], -5.0, 5.0))
b = tf.Variable(tf.random_uniform([1], -5.0, 5.0))

X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

hypothesis = W * X + b

cost = tf.reduce_mean(tf.square(hypothesis - Y))

a = tf.Variable(0.01)
optimizer = tf.train.GradientDescentOptimizer(a)
train = optimizer.minimize(cost)

init = tf.initialize_all_variables()

sess = tf.Session()
sess.run(init)

for step in xrange(2001):
    sess.run(train, feed_dict={X:x_data, Y:y_data})
    if step % 20 == 0:
        print step, sess.run(cost, feed_dict={X:x_data, Y:y_data}), sess.run(W), sess.run(b)

answer = sess.run(hypothesis, feed_dict={X:5})
print 'When X=5, hypothesis = ' + str(answer)

# Show the linear regression result
plt.figure(1)
plt.title('Linear Regression')
plt.xlabel('x')
plt.ylabel('y')

plt.plot(x_data, y_data, 'ro')

plt.plot(x_data, sess.run(W) * x_data + sess.run(b), 'b')

plt.plot([5], answer, 'go')
plt.show()

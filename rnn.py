import tensorflow as tf
import numpy as np

char_rdic = ['h', 'e', 'l', 'o']
char_dic = {w : i for i, w in enumerate(char_rdic)}
#print(char_dic)

sample = [char_dic[c] for c in 'hello']

#print(sample)

x_data = np.array([[1,0,0,0],
				   [0,1,0,0],
				   [0,0,1,0],
				   [0,0,1,0],
				   [0,0,0,1]], dtype = 'f')

#Configuration
char_vocab_size = len(char_dic)
rnn_size = 5
time_step_size = 5
batch_size = 1

#RNN Model
rnn_cell = tf.nn.rnn_cell.BasicRNNCell(rnn_size)
state = tf.zeros([batch_size, rnn_cell.state_size])

#x_data를 time_step_size로 나누어서 셀을 구성
X_split = tf.split(0, time_step_size, x_data)
outputs, state = tf.nn.rnn(rnn_cell, X_split, state)


logits = tf.reshape(tf.concat(1, outputs), [-1, rnn_size])
targets = tf.reshape(sample[0:], [-1])
#print(logits)
weights = tf.ones([5 * batch_size])
#weights = tf.ones([len(char_dic) * batch_size])

#예측값, 실제값, 비율

loss = tf.nn.seq2seq.sequence_loss_by_example([logits], [targets], [weights])

cost = tf.reduce_sum(loss) / batch_size
train_op = tf.train.RMSPropOptimizer(0.01, 0.9).minimize(cost)

#Launch the graph in a seesion
with tf.Session() as sess:
	tf.initialize_all_variables().run()
	for i in range(100) :
		sess.run(train_op)
		result = sess.run(tf.argmax(logits, 1))
		print(result)

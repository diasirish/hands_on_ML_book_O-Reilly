# Расчет идет по классической линейной регрессии 


import sklearn
import tensorflow as tf
import numpy as np
from sklearn.datasets import fetch_california_housing

housing = fetch_california_housing()
m, n = housing.data.shape
housing_data_plus_bias = np.c_[np.ones((m,1)), housing.data]
scaled_housing_data_plus_bias = sklearn.preprocessing.scale(housing_data_plus_bias, axis=0)

n_epochs = 1000
learning_rate = 0.01

X = tf.constant(scaled_housing_data_plus_bias, \
	dtype=tf.float32, name='X')
y = tf.constant(housing.target.reshape(-1,1), \
	dtype=tf.float32, name='y')
theta = tf.Variable(tf.random_uniform([n+1,1], -1.0, -1.0),\
	name='theta')
y_pred = tf.matmul(X, theta, name='predictions')
error = y_pred-y
mse = tf.reduce_mean(tf.square(error), name = 'mse')
gradients = 2/m*tf.matmul(tf.transpose(X), error)
training_op = tf.assign(theta, theta - learning_rate*gradients)

init = tf.global_variables_initializer()

with tf.Session() as sess:
	sess.run(init)
	
	for epoch in range(n_epochs):
		if epoch % 100 == 0:
			print("Эпоха", epoch, "MSE =", mse.eval())
		sess.run(training_op)
	best_theta = theta.eval()

	#print(best_theta)
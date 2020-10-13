# Расчет идет по классической линейной регрессии

# установка всех нужных пакетов
import sklearn
import tensorflow as tf
import numpy as np
from sklearn.datasets import fetch_california_housing
from datetime import datetime

# создаем отметки времени для журнального каталога
now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
root_logdir = "tf_logs"
logdir = "{}/run-{}/".format(root_logdir, now)

# загрузка нужных данных с которыми мы будем работать
housing = fetch_california_housing()
m, n = housing.data.shape
housing_data_plus_bias = np.c_[np.ones((m,1)), housing.data]
# для скорости градиентного вычесления мы стандартизируем данные
scaled_housing_data_plus_bias = sklearn.preprocessing.scale(housing_data_plus_bias, axis=0)



### СТАДИЯ ПОСТРОЕНИЯ  ####

# задаем параметры к обучению
n_epochs = 1000
learning_rate = 0.01

#определяем X и y для создания мини-пакетного градиентного спуска
#создание узлов-заполнителей
X = tf.placeholder(tf.float32, shape=(None, n+1), name="X")
y = tf.placeholder(tf.float32, shape=(None,1), name="y")

#определяем размер пакета и подсчитываем количество пакетов
batch_size = 100
n_batches = int(np.ceil(m/batch_size))

# выгрузка данных с диска (Не в книге, стырил с https://github.com/ageron/handson-ml/blob/master/09_up_and_running_with_tensorflow.ipynb)
def fetch_batch(epoch, batch_index, batch_size):
	np.random.seed(epoch * n_batches + batch_index)
	indices = np.random.randint(m, size=batch_size)
	X_batch = scaled_housing_data_plus_bias[indices]
	y_batch = housing.target.reshape(-1, 1)[indices]
	return X_batch, y_batch

#создаем переменную VARIABLE theta которую будем усовершенствовать
#насколько я понимаю по созданию theta как varible наши оптимизаторы
#будут знать, что нужно делать градиентный спуск именно по theta
theta = tf.Variable(tf.random_uniform([n+1,1], -1.0, -1.0),\
	name='theta')
# y_hat=X*theta
y_pred = tf.matmul(X, theta, name='predictions')
# ошибка и ошибка mse с функцией reduce_mean
error = y_pred-y
mse = tf.reduce_mean(tf.square(error), name = 'mse')

# Либо еще проще, используем оптимизатор
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
training_op = optimizer.minimize(mse)

# эта функция создает нам все переменные заданные выше
init = tf.global_variables_initializer()

# создаем узел saver
saver = tf.train.Saver()

#создание узла сводки (summary) или двоичного журнала для TensorBoard
mse_summary = tf.summary.scalar('MSE', mse)
file_writer = tf.summary.FileWriter(logdir, tf.get_default_graph())



### СТАДИЯ ВЫПОЛНЕНИЯ ###

# создаем сессию внутри блока with (не нужно sess.close())
with tf.Session() as sess:
	# действительная инициализация всех переменных
	sess.run(init)

	# 1000 раз запускаем узел training_op
	for epoch in range(n_epochs):
		for batch_index in range(n_batches):
			X_batch, y_batch = fetch_batch(epoch, batch_index, batch_size)
			# чек-поинт, на каждой 100 эпохе создаем
			#if epoch % 100 == 0:
				# сохраняем сеанс и модель
			#	save_path = saver.save(sess, "/tmp/my_model.ckpt")
			#	print("Эпоха", epoch, "MSE =", mse.eval())
			# чек-поинт, на каждой 10ой эпохе
			if epoch % 10 == 0:
				summary_str = mse_summary.eval(feed_dict = {X: X_batch, y: y_batch})
				step = epoch * n_batches + batch_index
				file_writer.add_summary(summary_str, step)
			# тут мы гоняем mse по плоскости, пытаясь найти min
			# и вставляем X_batch в placeholder X через параметер feed_dict
			sess.run(training_op, feed_dict = {X: X_batch, y: y_batch})
	# сохроняем лучшее значение и нашу модель
	best_theta = theta.eval()
	save_path = saver.save(sess, "/tmp/my_model_final.ckpt")
	print(best_theta)
	file_writer.close()



### ЗАГРУЗКИ ###

# ~~~~ загружаем предыдущую модель вызывая метод restore объекта Saver
#with tf.Session() as sess:
#	saver.restore(sess, "tmp/my_model_final.ckpt")
#	...

# ~~~ использование meta файла для загрузки всей архитектуры графа без надобности
# ~~~ базового кода

#saver = tf.train.import_meta_graph("/tmp/my_model_final.ckpt.meta")

#with tf.Session() as sess:
#	saver.restore(sess, "/tmp/my_model_final.ckpt")
#	...



### TENSOR BOARD ###

# как запустить tensorboard с terminal
# $ tensorboard --logdir tf_logs/

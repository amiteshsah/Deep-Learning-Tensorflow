# -*- coding: utf-8 -*-
# X is the number of fires 
# Y is the number of thefts, then: Y = f(X) , Y= WX + b (linear model)
# Model => Y= WX + b  ,   Loss= >  (Y - Y_predicted)^2

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import xlrd

DATA_FILE='/Users/amitesh/Desktop/CNN/tensor_code/stanford_git/data/fire_theft.xls'

# Step 1: read in data from the .xls file
book = xlrd.open_workbook(DATA_FILE, encoding_override="utf-8")
sheet = book.sheet_by_index(0)
data = np.asarray([sheet.row_values(i) for i in range(1, sheet.nrows)])
n_samples = sheet.nrows - 1

# Step 2: create placeholders for input X (number of fire) and label Y (number of theft)
X = tf.placeholder(tf.float32, name='X')
Y = tf.placeholder(tf.float32, name='Y')

# Step 3: create weight and bias, initialized to 0
w = tf.Variable(0.0, name='weights')
b = tf.Variable(0.0, name='bias')

# Step 4: build model to predict Y
Y_predicted = X * w + b 

# Step 5: use the square error as the loss function
loss = tf.square(Y - Y_predicted, name='loss')
# loss = utils.huber_loss(Y, Y_predicted)

# Step 6: using gradient descent with learning rate of 0.01 to minimize loss
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001).minimize(loss)
# Other optimizers
# tf.train.GradientDescentOptimizer
# tf.train.AdagradOptimizer
# tf.train.MomentumOptimizer
# tf.train.Adamoptimizer
# tf.train.ProximalGradientDescentOptimizer
# tf.train.ProximalAdagradOptimizer
# tf.train.RMSPropOptimizer # Geoff hilton class
with tf.Session() as sess:
	# Step 7: initialize the necessary variables, in this case, w and b
	sess.run(tf.global_variables_initializer()) 
	writer = tf.summary.FileWriter('./graphs/03/linear_reg', sess.graph)
	
	# Step 8: train the model
	for i in range(100): # train the model 100 epochs
		total_loss = 0
		for x, y in data:
			# Session runs train_op and fetch values of loss
			_, l = sess.run([optimizer, loss], feed_dict={X: x, Y:y}) 
			total_loss += l
		print('Epoch {0}: {1}'.format(i, total_loss/n_samples))
	# close the writer when you're done using it
	writer.close() 
	
	# Step 9: output the values of w and b
	w_value, b_value = sess.run([w, b]) 

# plot the results
X, Y = data.T[0], data.T[1]
plt.plot(X, Y, 'bo', label='Real data')
plt.plot(X, X * w_value + b_value, 'r', label='Predicted data')
plt.legend()
plt.show()

# Huber Loss = Robust to outliers. If the distance between the predicted value and the real value is small, square it . If its large, take its absolute value
# def huber_loss(labels,predictions,delta=1.0):
# 	residual1=tf.abs(predictions-labels)
# 	condition=tf.less(residual,delta)  # if residual < delta
# 	small_res=0.5*tf.square(residual)
# 	large_res=delta*residual-0.5*tf.square(delta)
# 	return tf.select(condition,small_res, large_res)  # if condition true, return small_res else return large_res
















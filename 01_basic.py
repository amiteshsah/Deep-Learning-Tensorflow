
import tensorflow as tf
# https://www.tensorflow.org/api_docs/python/

a= tf.constant(2, name="a")  # tf.constant(value, dtype=None, shape=None ,name='Const' , verify_shape=False)
b= tf.constant(3 , name="b")
#c=tf.constant(np.random.normal(size=(3,4)))
#z= tf.zeros([2,3],dtype=tf.int32, name="z" )
#z1 = tf.zeros_like(input_tensor)  # creates a tensor of shape and type of input tensor but all elements are zeros
x= tf.add(a,b , name="add")
y= tf.multiply(a,b , name="mul")
with tf.Session() as sess:
	#add this line to use TensorBoard
	writer= tf.summary.FileWriter("./graphs",sess.graph)
	#print sess.run(x)
	x,y =sess.run([x,y])
	print x,y
writer.close()

#---Go to terminal , run
# python [yourprogram].py
#---To view the graph, type in terminal
#tensorboard --logdir="./graphs" --port 6006
#https://localhost:6006/

#--- to start the session in terminal
#>>>tf.InteractiveSession()
#>>>x.eval()

#tf.ones(shape, dtype=tf.float32,name=None)
#tf.fills(dims,value,name=None)   # creates a tensor filled with scalar value

tf . linspace ( start ,  stop ,  num ,  name = None)
tf . linspace ( 10.0 ,   13.0 ,   4 ,  name = "linspace" )   ==>   [ 10.0   11.0   12.0   13.0]

tf . range ( start ,  limit = None ,  delta = 1 ,  dtype = None ,  name = 'range')
tf . range ( start ,  limit ,  delta )   ==>   [ 3 ,   6 ,   9 ,   12 ,   15]   ## 'start' is 3, 'limit' is 18, 'delta' is 3
tf . range ( limit )   ==>   [ 0 ,   1 ,   2 ,   3 ,   4]   # 'limit' is 5

tf.random_normal(shape, mean=0.0, stddev=1.0, dtype=tf.float32, seed=None, name=None) tf.truncated_normal(shape, mean=0.0, stddev=1.0, dtype=tf.float32, seed=None, name=None)
tf.random_uniform(shape, minval=0, maxval=None, dtype=tf.float32, seed=None, name=None)
tf.random_shuffle(value, seed=None, name=None)  # it will shuffle the images in first dimension i.e. row shuffling
tf.random_crop(value, size, seed=None, name=None)
tf.multinomial(logits, num_samples, seed=None, name=None) 
tf.random_gamma(shape, alpha, beta=None, dtype=tf.float32, seed=None, name=None)
tf.set_random_seed(seed)

  
a=tf.constants([3,6])
b=tf.constants([2,2])
tf.add(a,b)  # >> [5,8]
tf.add_n([a,b,b]) #>> [7 10] . Equivalent to a+b+b
tf.mul(a,b)  # Element wise multiplication
tf.matmul(a,b) # Matrix multiplication a=M*N , b=N*D , output= M*D
tf.matmul(tf.reshape(a,[1,2]),tf.reshape(b,[2,1]))
tf.divide(a , b) # >> [1,3]
tf.mod(a,b)  # [1 0]


print sess.graph.as_graph_def()  # It prints out the graph def


a  =  tf . Variable ( 2 ,  name = "scalar" )
b  =  tf . Variable ([ 2 ,   3 ],  name = "vector" )
c  =  tf . Variable ([[ 0 ,   1 ],   [ 2 ,   3 ]],  name = "matrix" )
W  =  tf . Variable ( tf . zeros ([ 784 , 10 ]))
#tf.Variable hold several ops
x = tf.Variable(...)
x.initializer # init 
x.value() # read op 
x.assign(...) # write op 
x.assign_add(...) # increment value 
x.assign_sub(...) # decrement value
init=tf.global_variables_initializer()
init_ab=tf.variables_initializer([a,b], name="init_ab")

W=tf.Variable(10)
assign_op=W.assign(100)
with tf.Session() as sess:
	sess.run(W.initializer)
	#print W.eval() #>> 10
	sess.run(assign_op)
print W.eval() # >> 100

W=tf.Variable(10)
sess1=tf.Session()
sess1.run(W.initializer)
print sess1.run(W.assign_add(10)) # >> 20
sess1.close()

tf.Graph.control_dependencies(control_inputs)

#f(x, y) = x*2 + y.
#x, y are placeholders for the actual values.
#With the graph assembled, we, or our clients, can later supply their own data when they need to execute the computation.
## create a placeholder of type float 32-bit, shape is a vector of 3 elements
a  =  tf . placeholder ( tf . float32 ,  shape =[ 3 ])
# create a constant of type float 32-bit, shape is a vector of 3 elements
b  =  tf . constant ([ 5 ,   5 ,   5 ],  tf . float32)
# use the placeholder as you would a constant or a variable
c  =  a  +  b  # Short for tf.add(a, b)
with  tf . Session ()   as  sess:
# feed [1, 2, 3] to placeholder a via the dict {a: [1, 2, 3]} # fetch value of c
print ( sess . run ( c ,   { a :   [ 1 ,   2 ,   3 ]}))  #>>   [ 6.   7.   8.]

# To feed multiple data points in . Wh feed all the values in , one at a time
with  tf . Session ()   as  sess:
	for  a_value  in  list_of_a_values: 
		print ( sess . run ( c ,   { a :  a_value }))


#You can feed values to tensors that aren’t placeholders. Any tensors that are feedable can be fed. To check if a tensor is feedable or not, use:
tf.Graph.is_feedable(tensor)
 
# create Operations, Tensors, etc (using the default graph) a = tf.add(2, 5)
b = tf.mul(a, 3)
# start up a `Session` using the default graph 
sess = tf.Session()
# define a dictionary that says to replace the value of `a` with 15 
replace_dict = {a: 15}
# Run the session, passing in `replace_dict` as the value to `feed_dict` 
sess.run(b, feed_dict=replace_dict) # returns 45

 
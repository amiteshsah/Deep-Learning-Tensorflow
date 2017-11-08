# Convolutional Neural Network for classifying images in the CIFAR-10 data-set
# Convolutional layer -> Max Pool -> Convolutional Layer -> Max Pool -> Fully Connected -> Fully Connected layer -> Softmax Classifier

import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from sklearn.metrics import confusion_matrix
import time
from datetime import timedelta
import math
import os
import urllib.request
import tarfile
import zipfile

# Use PrettyTensor to simplify Neural Network construction.
import prettytensor as pt

# URL for the data-set on the internet.
data_url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"

# Directory where you want to download and save the data-set.
data_path = "data/CIFAR-10/"

# Width and height of each image.
img_size = 32

# Number of channels in each image, 3 channels: Red, Green, Blue.
num_channels = 3

# Length of an image when flattened to a 1-dim array.
img_size_flat = img_size * img_size * num_channels

# Number of classes.
num_classes = 10

# Number of files for the training-set i.e. batches.
_num_files_train = 5

# Number of images for each batch-file in the training-set.
_images_per_file = 10000

# Total number of images in the training-set.
# This is used to pre-allocate arrays for efficiency.
_num_images_train = _num_files_train * _images_per_file

#The images are 32 x 32 pixels, but we will crop the images to 24 x 24 pixels.
img_size_cropped = 24

# Download the file from the internet.
def _print_download_progress(count, block_size, total_size):
    # Percentage completion.
    pct_complete = float(count * block_size) / total_size

    # Status-message. Note the \r which means the line should overwrite itself.
    msg = "\r- Download progress: {0:.1%}".format(pct_complete)

    # Print it.
    sys.stdout.write(msg)
    sys.stdout.flush()
filename = data_url.split('/')[-1]
file_path = os.path.join(data_path, filename)
file_path, _ = urllib.request.urlretrieve(url=data_url,
                                          filename=file_path,
                                          reporthook=_print_download_progress)

# Unpack the tar-ball.
tarfile.open(name=file_path, mode="r:gz").extractall(data_path)

# Load the names for the classes in the CIFAR-10 data-set
file_path = os.path.join(data_path, "cifar-10-batches-py/", "batches.meta")
with open(file_path, mode='rb') as file:
        data = pickle.load(file, encoding='bytes')
raw = data[b'label_names']
class_names= [x.decode('utf-8') for x in raw]

# Load the training-set. This returns the images, the class-numbers as integers, and the class-numbers as One-Hot encoded arrays called labels.

def load_training_data():
    # Pre-allocate the arrays for the images and class-numbers for efficiency.
    images = np.zeros(shape=[_num_images_train, img_size, img_size, num_channels], dtype=float)
    cls = np.zeros(shape=[_num_images_train], dtype=int)
    # Begin-index for the current batch.
    begin = 0
    # For each data-file.
    for i in range(_num_files_train):
        # Load the images and class-numbers from the data-file.
        images_batch, cls_batch = _load_data(filename="data_batch_" + str(i + 1))
        # Number of images in this batch.
        num_images = len(images_batch)
        # End-index for the current batch.
        end = begin + num_images
        # Store the images into the array.
        images[begin:end, :] = images_batch
        # Store the class-numbers into the array.
        cls[begin:end] = cls_batch
        # The begin-index for the next batch is the current end-index.
        begin = end

    return images, cls, np.eye(num_classes, dtype=float)[cls]

def _load_data(filename):
	file_path = os.path.join(data_path, "cifar-10-batches-py/", filename)
    print("Loading data: " + file_path)
    with open(file_path, mode='rb') as file:
        data = pickle.load(file, encoding='bytes')
    # Get the raw images.
    raw_images = data[b'data']
    # Get the class-numbers for each image. Convert to numpy-array.
    cls = np.array(data[b'labels'])

    # Convert the images.
    raw_float = np.array(raw_images, dtype=float) / 255.0
    # Reshape the array to 4-dimensions.
    images = raw_float.reshape([-1, num_channels, img_size, img_size])
    # Reorder the indices of the array.
    images = images.transpose([0, 2, 3, 1])
    return images, cls

images_train, cls_train, labels_train = load_training_data()

# Load the test test.
images_test, cls_test = _load_data(filename="test_batch")
labels_test = np.eye(num_classes, dtype=float)[cls_test]


#--------- Tensorflow Graph creation --------------

# placeholder variable for the input images. 
x = tf.placeholder(tf.float32, shape=[None, img_size, img_size, num_channels], name='x')


# placeholder variable for the true labels associated with the images that were input in the placeholder variable x. Each label is a vector of length num_classes
y_true = tf.placeholder(tf.float32, shape=[None, num_classes], name='y_true')

# placeholder variable for the class-number. It is maximum of each label vector that was hot encoded.
y_true_cls = tf.argmax(y_true, dimension=1) 

def pre_process_image(image, training):  
    if training:
        # Randomly crop the input image.
        image = tf.random_crop(image, size=[img_size_cropped, img_size_cropped, num_channels])

        # Randomly flip the image horizontally.
        image = tf.image.random_flip_left_right(image)
        
        # Randomly adjust hue, contrast and saturation.
        image = tf.image.random_hue(image, max_delta=0.05)
        image = tf.image.random_contrast(image, lower=0.3, upper=1.0)
        image = tf.image.random_brightness(image, max_delta=0.2)
        image = tf.image.random_saturation(image, lower=0.0, upper=2.0)

        # Limit the image pixels between [0, 1] in case of overflow.
        image = tf.minimum(image, 1.0)
        image = tf.maximum(image, 0.0)
    else:
        # Crop the input image around the centre so it is the same size as images that are randomly cropped during training.
        image = tf.image.resize_image_with_crop_or_pad(image,
                                                       target_height=img_size_cropped,
                                                       target_width=img_size_cropped)
    return image

# The function pre_preocess_image is called for each image in the input batch using the following function
def pre_process(images, training):
    # Use TensorFlow to loop over all the input images and call the function above which takes a single image as input.
    images = tf.map_fn(lambda image: pre_process_image(image, training), images)
    return images

images = pre_process(images=x, training=training)

# ---- some helper-functions which will be used several times in the graph construction -------
def new_weights(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.05))

def new_biases(length):
    return tf.Variable(tf.constant(0.05, shape=[length]))

def new_conv_layer(input,              # The previous layer.
                   num_input_channels, # Num. channels in prev. layer.
                   filter_size,        # Width and height of filters.
                   num_filters,        # Number of filters.
                   use_pooling=True):  # Use 2x2 max-pooling.
    # Shape of the filter-weights for the convolution.
    shape = [filter_size, filter_size, num_input_channels, num_filters]
    # Create new weights aka. filters with the given shape.
    weights = new_weights(shape=shape)
    # Create new biases, one for each filter.
    biases = new_biases(length=num_filters)

    # convolution operation
    # Note the strides are set to 1 in all dimensions. The first and last stride must always be 1, because the first is for the image-number and
    # the last is for the input-channel.
    # But e.g. strides=[1, 2, 2, 1] would mean that the filter
    # is moved 2 pixels across the x- and y-axis of the image.
    # The padding is set to 'SAME' which means the input image
    # is padded with zeroes so the size of the output is the same.
    layer = tf.nn.conv2d(input=input,
                         filter=weights,
                         strides=[1, 1, 1, 1],
                         padding='SAME')

    # Add the biases to the results of the convolution.
    layer += biases

    # Use pooling to down-sample the image resolution?
    if use_pooling:
        # This is 2x2 max-pooling, which means that we consider 2x2 windows and select the largest value
        # in each window. Then we move 2 pixels to the next window.
        layer = tf.nn.max_pool(value=layer,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding='SAME')

    # Rectified Linear Unit (ReLU). It calculates max(x, 0) for each input pixel x.
    # This adds some non-linearity to the formula and allows us to learn more complicated functions.
    layer = tf.nn.relu(layer)    # Note that ReLU is normally executed before the pooling.

    #We return both the resulting layer and the filter-weights
    return layer, weights

def flatten_layer(layer):
    # Get the shape of the input layer. Layer_shape == [num_images, img_height, img_width, num_channels]
    layer_shape = layer.get_shape()

    # The number of features is: img_height * img_width * num_channels
    num_features = layer_shape[1:4].num_elements()

    # Reshape the layer to [num_images, num_features].
    layer_flat = tf.reshape(layer, [-1, num_features])

    # Return both the flattened layer and the number of features.
    return layer_flat, num_features

def new_fc_layer(input,          # The previous layer.
                 num_inputs,     # Num. inputs from prev. layer.
                 num_outputs,    # Num. outputs.
                 use_relu=True): # Use Rectified Linear Unit (ReLU)?

    # Create new weights and biases.
    weights = new_weights(shape=[num_inputs, num_outputs])
    biases = new_biases(length=num_outputs)

    layer = tf.matmul(input, weights) + biases

    # Use ReLU?
    if use_relu:
        layer = tf.nn.relu(layer)

    return layer

# ----------- CNN Architecture --------------------
# First convolutional layer.
layer_conv1, weights_conv1 = new_conv_layer(input=images,
                   							num_input_channels=num_channels,
                   							filter_size=5,
                   							num_filters=16,
                   							use_pooling=True)

# Second convolutional layer.
layer_conv2, weights_conv2 = new_conv_layer(input=layer_conv1,
                   							num_input_channels=16,
                   							filter_size=5,
                   							num_filters=36,
                   							use_pooling=True)

# Flatten layer.
layer_flat, num_features = flatten_layer(layer_conv2)

# First fully-connected layer.
layer_fc1 = new_fc_layer(input=layer_flat,
                         num_inputs=num_features,
                         num_outputs=128,
                         use_relu=True)

# Second fully-connected layer.
layer_fc2 = new_fc_layer(input=layer_fc1,
                         num_inputs=128,
                         num_outputs=num_classes,
                         use_relu=False)

# Predicted class-label.
y_pred = tf.nn.softmax(layer_fc2)

# Cross-entropy for the classification of each image.
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=layer_fc2,
                                            			labels=y_true)

# Loss or cost measure is the scalar value that must be minimized.
loss = tf.reduce_mean(cross_entropy)

# AdamOptimizer to minimize the loss
optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(loss)

# The output of the network y_pred is an array with 10 elements. The class number is the index of the largest element in the array.
y_pred_cls = tf.argmax(y_pred, dimension=1)

# creates a vector of booleans telling us whether the predicted class equals the true class of each image.
correct_prediction = tf.equal(y_pred_cls, y_true_cls)

# Type cast bool to float. False becomes 0 and True becomes 1, and then take the average of these numbers.
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# To save the variables of the neural network
saver = tf.train.Saver()

# -------------- Tensorflow RUN -----------------------------
# create a TensorFlow session which is used to execute the graph
session = tf.Session()

# save checkpoints during training 
save_dir = 'checkpoints/'
os.makedirs(save_dir)
save_path = os.path.join(save_dir, 'cifar10_cnn')

# try to restore the latest checkpoint if present else initialize all the variables
try:
    print("Trying to restore last checkpoint ...")

    # Use TensorFlow to find the latest checkpoint - if any.
    last_chk_path = tf.train.latest_checkpoint(checkpoint_dir=save_dir)

    # Try and load the data in the checkpoint.
    saver.restore(session, save_path=last_chk_path)

    # If we get to this point, the checkpoint was successfully loaded.
    print("Restored checkpoint from:", last_chk_path)
except:
    # initialize all the variables for the TensorFlow graph.
    print("Failed to restore checkpoint. Initializing variables instead.")
    session.run(tf.global_variables_initializer())

# use a small batch of images in each iteration of the optimizer otherwise it takes a long time to calculate the gradient of the model using all these images
train_batch_size = 64

# In each iteration, a new batch of data is selected from the training-set and then TensorFlow executes the optimizer using those training samples. The progress is printed every 100 iterations.
def optimize(num_iterations):
    start_time = time.time()

    for i in range(num_iterations):
    	num_images = len(images_train)

		# Get a batch of training examples.
    	# Create a random index.
    	idx = np.random.choice(num_images,
                           size=train_batch_size,
                           replace=False)
    	# Use the random index to select random images and labels.
   		x_batch = images_train[idx, :, :, :]
    	y_batch = labels_train[idx, :]

        # Put the batch into a dict with the proper names for placeholder variables in the TensorFlow graph.
        feed_dict_train = {x: x_batch,
                           y_true: y_true_batch}

        # Run the optimizer using this batch of training data.
        session.run(optimizer, feed_dict=feed_dict_train)

        # Print status every 100 iterations.
        if i % 100 == 0:
            # Calculate the accuracy on the training-set.
            acc = session.run(accuracy, feed_dict=feed_dict_train)

            # Message for printing.
            msg = "Optimization Iteration: {0:>6}, Training Accuracy: {1:>6.1%}"
            print(msg.format(i + 1, acc))

    # Ending time.
    end_time = time.time()

    # Difference between start and end-times.
    time_dif = end_time - start_time

    # Print the time-usage.
    print("Time usage: " + str(timedelta(seconds=int(round(time_dif)))))

# Perform optimization
optimize(num_iterations=1000)

saver.save(session,save_path=save_path)
# ------- Prediction on Test Data ----------------
def predict_cls(images, labels, cls_true):
    # Number of images.
    num_images = len(images)
    # Preprocess on test images
    images = pre_process(images=images, training=False)
    # Allocate an array for the predicted classes
    cls_pred = np.zeros(shape=num_images, dtype=np.int)

    # Now calculate the predicted classes for the batches.
    # The starting index for the next batch is denoted i.
    i = 0

    while i < num_images:
        # The ending index for the next batch is denoted j.
        j = min(i + batch_size, num_images)

        # Create a feed-dict with the images and labels
        # between index i and j.
        feed_dict = {x: images[i:j, :],
                     y_true: labels[i:j, :]}

        # Calculate the predicted class using TensorFlow.
        cls_pred[i:j] = session.run(y_pred_cls, feed_dict=feed_dict)

        # Set the start-index for the next batch to the end-index of the current batch.
        i = j

    # Create a boolean array whether each image is correctly classified.
    correct = (cls_true == cls_pred)

    return correct, cls_pred

correct, cls_pred = predict_cls(images = images_test, labels = labels_test, cls_true = cls_test)
acc, num_correct =  correct.mean(), correct.sum()
# Number of images being classified.
num_images = len(correct)
# Print the accuracy.
msg = "Accuracy on Test-Set: {0:.1%} ({1} / {2})"
print(msg.format(acc, num_correct, num_images))

def plot_confusion_matrix(cls_pred):
    # Get the confusion matrix using sklearn.
    cm = confusion_matrix(y_true=cls_test,  # True class for test-set.
                          y_pred=cls_pred)  # Predicted class.

    # Print the confusion matrix as text.
    for i in range(num_classes):
        # Append the class-name to each line.
        class_name = "({}) {}".format(i, class_names[i])
        print(cm[i, :], class_name)

    # Print the class-numbers for easy reference.
    class_numbers = [" ({0})".format(i) for i in range(num_classes)]
    print("".join(class_numbers))
print("Confusion Matrix:")
plot_confusion_matrix(cls_pred=cls_pred)


def plot_images(images, cls_true, cls_pred=None, smooth=True):
    fig, axes = plt.subplots(3, 3)
    # Adjust vertical spacing if we need to print ensemble and best-net.
    if cls_pred is None:
        hspace = 0.3
    else:
        hspace = 0.6
    fig.subplots_adjust(hspace=hspace, wspace=0.3)

    for i, ax in enumerate(axes.flat):
        # Interpolation type.
        if smooth:
            interpolation = 'spline16'
        else:
            interpolation = 'nearest'

        # Plot image.
        ax.imshow(images[i, :, :, :],
                  interpolation=interpolation)
            
        # Name of the true class.
        cls_true_name = class_names[cls_true[i]]

        # Show true and predicted classes.
        if cls_pred is None:
            xlabel = "True: {0}".format(cls_true_name)
        else:
            # Name of the predicted class.
            cls_pred_name = class_names[cls_pred[i]]
            xlabel = "True: {0}\nPred: {1}".format(cls_true_name, cls_pred_name)
        # Show the classes as the label on the x-axis.
        ax.set_xlabel(xlabel)
        
        # Remove ticks from the plot.
        ax.set_xticks([])
        ax.set_yticks([])
    
    # Ensure the plot is shown correctly with multiple plots
    # in a single Notebook cell.
    plt.show()






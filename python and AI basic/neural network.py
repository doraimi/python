import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist

# Load the MNIST dataset
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# Normalize the pixel values to range [0, 1]
train_images, test_images = train_images / 255.0, test_images / 255.0

# Build the neural network model
model = models.Sequential([
    layers.Flatten(input_shape=(28, 28)),       # Flatten the 28x28 images to a 1D array
    layers.Dense(128, activation='relu'),      # Fully connected layer with 128 neurons and ReLU activation
    layers.Dropout(0.2),                        # Dropout layer to reduce overfitting
    layers.Dense(10, activation='softmax')      # Output layer with 10 neurons for 10 classes (digits 0-9) and softmax activation
])

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(train_images, train_labels, epochs=5)

# Evaluate the model
test_loss, test_acc = model.evaluate(test_images, test_labels)
print(f'Test accuracy: {test_acc}')

"""
Certainly! Here's a simple example code using Python and the popular deep learning library TensorFlow to create a basic neural network for classifying handwritten digits from the MNIST dataset:

This code performs the following steps:

Loading the MNIST dataset: The code loads the MNIST dataset containing grayscale images of handwritten digits along with their corresponding labels.

Preprocessing: It preprocesses the pixel values by scaling them to a range between 0 and 1.

Building the neural network model: It constructs a simple neural network model using the Sequential API provided by TensorFlow. The model consists of a Flatten layer to flatten the input images into a 1D array, a fully connected (Dense) layer with 128 neurons and ReLU activation function, a Dropout layer for regularization, and an output Dense layer with 10 neurons for classifying the digits (0-9) using softmax activation.

Compiling the model: The model is compiled with the Adam optimizer, sparse categorical crossentropy loss function, and accuracy metric.

Training the model: The model is trained on the training data for 5 epochs.

Evaluating the model: Finally, the model is evaluated on the test dataset, and the test accuracy is printed.

This example demonstrates a basic neural network for digit classification using TensorFlow.
"""
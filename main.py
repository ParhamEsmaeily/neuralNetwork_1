import tensorflow as tf
from tensorflow import keras
import numpy as np

# Define the training data
training_data = np.array([[1, 1, 1, 0, 0],
                          [1, 1, 1, 0, 1],
                          [0, 0, 0, 1, 0],
                          [0, 0, 0, 1, 1],
                          [0, 0, 1, 0, 0],
                          [0, 0, 1, 0, 1],
                          [0, 0, 1, 1, 0],
                          [0, 0, 1, 1, 1],
                          [0, 1, 0, 0, 0],
                          [0, 1, 0, 0, 1],
                          [0, 1, 0, 1, 0],
                          [0, 1, 0, 1, 1],
                          [0, 1, 1, 0, 0],
                          [0, 1, 1, 0, 1],
                          [0, 1, 1, 1, 0],
                          [0, 1, 1, 1, 1],
                          [1, 0, 0, 0, 0],
                          [1, 0, 0, 0, 1],
                          [1, 0, 0, 1, 0],
                          [1, 0, 0, 1, 1],
                          [1, 0, 1, 0, 0],
                          [1, 0, 1, 0, 1],
                          [1, 0, 1, 1, 0],
                          [1, 0, 1, 1, 1]])

target_data = np.array([[1],
                        [1],
                        [0],
                        [0],
                        [1],
                        [1],
                        [1],
                        [1],
                        [0],
                        [0],
                        [0],
                        [0],
                        [1],
                        [1],
                        [1],
                        [1],
                        [1],
                        [1],
                        [1],
                        [1],
                        [1],
                        [1],
                        [1],
                        [1]])

# Build the neural network model
#1 hidden layer or the first keras is known as the input layer
model = keras.Sequential([
    keras.layers.Dense(10, input_shape=(5,), activation='relu'),
    keras.layers.Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
# Train the model epoch = that the training data is fed to the model 1000 times
model.fit(training_data, target_data, epochs=1000, verbose=1)
# Testing data
testing_data = np.array([[1, 1, 0, 0, 0],
                         [1, 1, 0, 0, 1],
                         [0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 1],
                         [1, 1, 0, 1, 0],
                         [1, 1, 0, 1, 1],
                         [1, 1, 1, 1, 0],
                         [1, 1, 1, 1, 1]])

# Make predictions
predictions = model.predict(testing_data)
binary_predictions = [1 if predictions >= 0.70 else 0 for predictions in predictions]

# Print the predictions
for i in range(len(testing_data)):
    print("Input:", testing_data[i])
    print("Prediction:", predictions[i])
    print("Binary:", binary_predictions[i])


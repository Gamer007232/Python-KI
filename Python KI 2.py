#in diesem Abschnitt beginnen wir mit dem Einladen der Libraries
import tensorflow as tf 
import numpy as np

#in diesem Abschnitt laden wir die Daten (Bilder und Labels) aus der Datenbank "Mnist Fashion" ein
mnist = tf.keras.datasets.fashion_mnist
(training_images, training_labels), (test_images, test_labels) = mnist.load_data()

#in diesem Abschnitt formen wir unsere Daten so um, dass unser neuronales Netz mit den Daten arbeiten kann
training_images = training_images.reshape(60000, 28, 28, 1)
training_images = training_images / 255.0
test_images = test_images.reshape(10000, 28, 28, 1)
test_images = test_images / 255.0

#in diesem Abschnitt "bauen" wir uns neuronales Netz
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

#in diesem Abschnitt trainieren wir unser neuronales Netz
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(training_images,training_labels, epochs=4)


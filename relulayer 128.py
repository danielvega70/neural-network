import tensorflow as tf

# Define the input shape
input_shape = 10 # assuming there are 10 features in the input data

# Define the model architecture
model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, input_shape=(input_shape,)),
    tf.keras.layers.ReLU(),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=32)

# Evaluate the model on the test data
test_loss, test_acc = model.evaluate(X_test, y_test)
print('Test accuracy:', test_acc)

# lab6_cc3_ver3
# Coding Challenege 3: Sentiment Analysis with Python Tensorflow
# Dwyer Bradley

"""
Sentiment analysis (SA) model made with tensorflow using IMDB reviews for training and test
Model saved with pickle to be used for SA in other programs
"""

# Import packages
import tensorflow as tf
from tensorflow import keras
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import pickle

# Print a message for package installation
print("Packages imported successfully :)")

##################

### Testing on IMDB dataset ###

# Load the IMDB dataset
(train_data, test_data), info = tfds.load(
    'imdb_reviews/subwords8k',
    split=(tfds.Split.TRAIN, tfds.Split.TEST),
    with_info=True,
    as_supervised=True
)

# Tokenize reviews
tokenizer = info.features['text'].encoder

train_data = train_data.shuffle(10000).padded_batch(64, padded_shapes=([None], []))
test_data = test_data.padded_batch(64, padded_shapes=([None], []))

# Build model
model = keras.Sequential([
    keras.layers.Embedding(tokenizer.vocab_size, 64),
    keras.layers.Bidirectional(keras.layers.LSTM(64)),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid')
])

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train model
history = model.fit(
    train_data,
    epochs=10,
    validation_data=test_data
)

# Evaluate model
def plot_graphs(history, string):
    plt.plot(history.history[string])
    plt.plot(history.history['val_' + string])
    plt.xlabel('Epochs')
    plt.ylabel(string)
    plt.legend([string, 'val_' + string])
    plt.show()

plot_graphs(history, 'accuracy')
plot_graphs(history, 'loss')

# Make predictions
# Try example prediction
sample_text = "I love this movie!"
sample_text = tokenizer.encode(sample_text)
sample_text = pad_sequences([sample_text], maxlen=64, padding='post', truncating='post')
prediction = model.predict(sample_text)

print(f"Prediction: {prediction[0][0]}")

with open('model.pkl', 'wb') as SA_IMDB_model_file :
    pickle.dump(model, SA_IMDB_model_file)
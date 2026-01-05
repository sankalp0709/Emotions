import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import h5py

weights_file = 'model2.weights.h5'

def create_model():
    model = Sequential()
    model.add(Dense(128, activation='relu', input_shape=(10,)))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(2, activation='softmax'))
    return model

print("Trying load_weights...")
try:
    model = create_model()
    model.load_weights(weights_file)
    print("Success load_weights")
except Exception as e:
    print(f"Failed load_weights: {e}")

print("\nTrying load_weights with by_name=True...")
try:
    model = create_model()
    model.load_weights(weights_file, by_name=True)
    print("Success load_weights by_name=True")
except Exception as e:
    print(f"Failed load_weights by_name: {e}")

print("\nTrying load_model...")
try:
    model = tf.keras.models.load_model(weights_file)
    print("Success load_model")
except Exception as e:
    print(f"Failed load_model: {e}")

import numpy as np
import pandas as pd
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import LSTM, Dense, GRU, Embedding
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
import utility
import tensorflow as tf

GRAMS = 10

def encode(grams, raw_data):
    chars = sorted(list(set(raw_data)))
    mapping = dict((c, i) for i, c in enumerate(chars))

    sequences = list()
    for line in grams:
        # integer encode line
        encoded_seq = [mapping[char] for char in line]
        # store
        sequences.append(encoded_seq)
    return sequences, mapping


def build_model(vocab):
    model = Sequential()
    model.add(Embedding(vocab, 25, input_length=GRAMS-1, trainable=True))
    model.add(GRU(50, recurrent_dropout=0.1, dropout=0.1))
    model.add(Dense(vocab, activation='softmax'))

    model.compile(loss='categorical_crossentropy', metrics=['acc'], optimizer='adam')

    return model


def main():
    data, hold_out = utility.read_file('cwe-train.txt')

    n_grams = utility.gen_n_grams(data, GRAMS)
    n_grams, mapping = encode(n_grams, data)

    vocab = len(mapping)
    sequences = np.array(n_grams)
    # create X and y
    x, y = sequences[:, :-1], sequences[:, -1]
    # one hot encode y
    y = to_categorical(y, num_classes=vocab)
    # create train and validation sets
    x_tr, x_val, y_tr, y_val = train_test_split(x, y, test_size=0.1, random_state=42)

    print('Train shape:', x_tr.shape, 'Val shape:', x_val.shape)

    model = build_model(vocab)

    stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=5)

    model.fit(x_tr, y_tr, epochs=100, verbose=1, validation_data=(x_val, y_val), callbacks=stop_early)

if __name__ == '__main__':
    main()
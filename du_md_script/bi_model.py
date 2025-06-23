from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, LayerNormalization, BatchNormalization, MaxPooling1D, Flatten, LSTM, Dense, ReLU, Dropout, Bidirectional
from tensorflow.keras import layers, models

def build_model(input_shape, num_classes):
    model = Sequential()

    # Block 1: Conv1D + MaxPooling1D
    model.add(Conv1D(filters=128, kernel_size=5, strides=1, activation='relu', input_shape=input_shape, padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(pool_size=2))  # seq_length = 90 -> 45

    # Block 2: Conv1D
    model.add(Conv1D(filters=256, kernel_size=3, strides=2, activation='relu',
                     padding='same'))
    model.add(BatchNormalization())
    # model.add(MaxPooling1D(pool_size=2))

    # Block 3: Conv1D
    model.add(Conv1D(filters=512, kernel_size=3, strides=1, activation='relu', padding='same'))
    model.add(BatchNormalization())
    # model.add(MaxPooling1D(pool_size=2))

    # BiLSTM
    model.add(Bidirectional(LSTM(128, return_sequences=False)))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))

    # Output layer
    model.add(Dense(num_classes, activation='sigmoid'))

    return model

# def build_model(input_shape, num_classes):
#     model = Sequential()
#
#     # Conv1D
#     model.add(Conv1D(filters=512, kernel_size=3, strides=1, activation='relu', input_shape=input_shape, padding='same'))
#     model.add(LayerNormalization())
#     model.add(MaxPooling1D(pool_size=2))
#     #
#     # # Conv1D
#     model.add(Conv1D(filters=256, kernel_size=3, strides=2, activation='relu'))
#     model.add(LayerNormalization())
#     model.add(MaxPooling1D(pool_size=2))
#
#     # Conv1D
#     model.add(Conv1D(filters=128, kernel_size=3, strides=2, activation='relu'))
#     model.add(LayerNormalization())
#     # model.add(MaxPooling1D(pool_size=2))
#
#     # BiLSTM
#     model.add(Bidirectional(LSTM(256, return_sequences=False, input_shape=input_shape)))
#     model.add(LayerNormalization())
#     model.add(Dropout(0.5))
#
#     # BiLSTM
#     # model.add(Bidirectional(LSTM(128, return_sequences=False)))
#     # model.add(LayerNormalization())
#     # model.add(Dropout(0.5))
#
#     # Dense layer
#     model.add(Dense(128))
#     model.add(LayerNormalization())
#     model.add(ReLU())
#     model.add(Dropout(0.5))
#     #
#     # # Output layer
#     model.add(Dense(num_classes, activation='sigmoid'))
#
#     return model

def build_model_cnn(input_shape, num_classes):
    model = Sequential()

    model.add(Conv1D(filters=128, kernel_size=5, strides=1, activation='relu', input_shape=input_shape, padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(pool_size=2))

    model.add(Conv1D(filters=256, kernel_size=3, strides=2, activation='relu', padding='same'))
    model.add(BatchNormalization())

    model.add(Conv1D(filters=512, kernel_size=3, strides=1, activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(pool_size=2))  # Optional

    model.add(Flatten())
    model.add(Dense(128))
    model.add(BatchNormalization())
    model.add(ReLU())
    model.add(Dropout(0.5))

    model.add(Dense(num_classes, activation='sigmoid'))

    return model


def build_model_lstm(input_shape, num_classes):
    model = Sequential()

    model.add(LSTM(256, input_shape=input_shape, return_sequences=True))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))

    model.add(LSTM(128, input_shape=input_shape, return_sequences=False))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))

    model.add(Dense(num_classes, activation='sigmoid'))

    return model


def build_model_bilstm(input_shape, num_classes):
    model = Sequential()

    model.add(Bidirectional(LSTM(256, return_sequences=True), input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))

    model.add(Bidirectional(LSTM(128, return_sequences=False), input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))

    # model.add(Dense(128))
    # model.add(BatchNormalization())
    # model.add(ReLU())
    # model.add(Dropout(0.5))

    model.add(Dense(num_classes, activation='sigmoid'))

    return model



import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense, LSTM, Dropout
import matplotlib.pyplot as plt
# %matplotlib inline

def RNN_Model(train_x, train_y):
    model = Sequential() 
    model.add(SimpleRNN(units=32, input_shape=(1, 5), activation='relu')) 
    model.add(Dense(8, activation='relu'))
    model.add(Dense(1))
    model.compile(loss='mse', optimizer='adam', metrics=['acc'])
    print(model.summary())

    history=model.fit(train_x, train_y, epochs=10)
    return history


def plt_History(history):



    plt.plot(history.history['loss'], label='loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(loc='best')
    plt.show()
    return plt.show()


def plt_Predict(test_x, test_y):
    pred_y = model.predict(test_x)

    plt.figure()
    plt.plot(test_y, color='red', label='real target y')
    plt.plot(pred_y, color='blue', label='predict y')
    plt.legend()
    plt.show()
    return pred_y, plt.show()
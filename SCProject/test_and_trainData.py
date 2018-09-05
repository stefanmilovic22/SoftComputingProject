from keras.utils import to_categorical
from keras.datasets import mnist
from nn_model import srediSliku, neural_network_model


#Main funkcija za pokretanje programa (treniranje podataka)
def main():

    (train_pictures, train_labels), (test_pictures, test_labels) = mnist.load_data()

    # (0-9)
    class_num = 10
    j = 1
    maxValue = 255
    fl = 'float32'

    for index in range(len(test_pictures)):
        cropped = srediSliku(test_pictures[index])
        test_pictures[index] = cropped

    for index in range(len(train_pictures)):
        cropped = srediSliku(train_pictures[index])
        train_pictures[index] = cropped

    row, col = train_pictures.shape[j:]

    train_data = train_pictures.reshape(train_pictures.shape[0], row, col, j)
    test_data = test_pictures.reshape(test_pictures.shape[0], row, col, j)
    shape = (row, col, j)

    train_data = train_data.astype(fl)
    test_data = test_data.astype(fl)

    # Scale the data to lie between 0 to 1
    train_data /= maxValue
    test_data /= maxValue

    # konverzija iz tipa int u categorical
    train_lab_categorical = to_categorical(train_labels)
    test_lab_categorical = to_categorical(test_labels)

    model = neural_network_model(shape, class_num)

    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

    model.summary()

    #istorija
    hist = model.fit(train_data, train_lab_categorical, batch_size=256, epochs=30, verbose=1,
                         validation_data=(test_data, test_lab_categorical))
    loss, accuracy = model.evaluate(test_data, test_lab_categorical, verbose=0)  # racunamo gubitke i tacnost

    model.save_weights('neuralModel.h5')

    print(accuracy)



if __name__ == "__main__":
    main()
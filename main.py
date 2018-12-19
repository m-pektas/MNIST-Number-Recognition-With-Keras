#import libraries
import keras
from keras.models import Sequential, load_model
from keras.layers import MaxPooling2D, Conv2D, Dense, Dropout, Flatten
from keras.datasets import mnist
import matplotlib.pyplot as plt

#load and split data
(X_train,Y_train), (x_test, y_test) = mnist.load_data()


#4 image show
plt.figure(figsize=(14,14))
for i in range(4):
    plt.imshow(X_train[i])
    plt.show()


#set parameters
batch_size = 128
num_classes = 10
epochs = 1

#image size
img_rows, img_cols = 28,28

X_train = X_train.reshape(X_train.shape[0],img_rows, img_cols, 1)
x_test = x_test.reshape(x_test.shape[0],img_rows, img_cols, 1)
input_dim = (img_rows, img_cols,1)

#one hot encoding
Y_train = keras.utils.to_categorical(Y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)


#create model
model = Sequential()
model.add(Conv2D(32,kernel_size=(3,3), activation= 'relu', input_shape=input_dim))
model.add(Conv2D(64, kernel_size=(3,3), activation= 'relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation= 'relu'))
model.add(Dropout(0.50))
model.add(Dense(num_classes, activation= 'softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.fit(X_train, Y_train, batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(x_test, y_test))

#save model
model.save('./models/mnist_number_model.h5')

#test model
score = model.evaluate(x_test,y_test,verbose=0)
print('Test Loss:',score[0])
print('Test Accuracy :',score[1])


#1 image test with model
test_image = x_test[66]
plt.imshow(test_image.reshape(28,28))
plt.show()

test_data = x_test[66].reshape(1,28,28,1)
pred = model.predict(test_data,batch_size=1)
print(pred)
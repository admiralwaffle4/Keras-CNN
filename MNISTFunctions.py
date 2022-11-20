import numpy as np
import pandas as pd

data = pd.read_csv("mnist_train.csv") #this dataset is from Kaggle and contains 60,000 images of handwritten digits; pandas just lets me read it

data = np.array(data) #convert the data to a numpy array, so we can actually use it

np.random.shuffle(data) #shuffle it so its different every time
m, n = data.shape #get the amount of rows (m: images) and columns (n: pixels + label)

data_dev = data[:10000].T #take the first 10,000 images for development
Y_dev = data_dev[0] # flip it for reasons
X_dev = data_dev[1:n] #get the rest of the data
X_dev /= np.int64(255) #normalize the data

data_train = data[10000:m].T #take the rest of the images for training
Y_train = data_train[0] # flip it for reasons
X_train = data_train[1:n] #get the rest of the data
X_train /= np.int64(255) #normalize the data
_,m_train = X_train.shape #grab the size of the training data

#initialize the weights and biases randomly
def init_params():
    W1 = np.random.rand(10, 784) - 0.5 #initialize the weights for the first layer; will be changed later bc learning
    b1 = np.random.rand(10, 1) - 0.5 #initialize the biases for the first layer; will be changed later bc learning

    W2 = np.random.rand(10, 10) - 0.5 #initialize the weights for the second layer; will be changed later bc learning
    b2 = np.random.rand(10, 1) - 0.5 #initialize the biases for the second layer; will be changed later bc learning

    return W1, b1, W2, b2

#ReLU activation function: if the input is greater than 0, return the input; otherwise, return 0
def ReLU(Z):
    return  np.maximum(0, Z)

#derivative of the ReLU activation function
def dReLU(Z):
    return Z > 0

#softmax classification: for each row, divide each element by the sum of the row
def softmax(Z):
    exp = np.exp(Z - np.max(Z))
    return exp / exp.sum(axis=0)

#one-hot encoding: for each row, set the element at the index of the label to 1 and the rest to 0
def one_hot(Y):
    Y_one_hot = np.zeros((Y.size, Y.max()+1)) #create a matrix of zeros with the same amount of rows as Y and 10 columns
    Y_one_hot[np.arange(Y.size), Y] = 1 #for each row, go to the column that matches the label and set it to one
    Y_one_hot = Y_one_hot.T
    return Y_one_hot

#forward propagation: calculate the output of each layer
def forward_prop(W1, b1, W2, b2, X):
    Z1 = W1.dot(X) + b1 #calculate the weighted sum of the inputs for the first layer
    A1 = ReLU(Z1) #calculate the activation of the first layer

    Z2 = W2.dot(A1) + b2 #calculate the weighted sum of the inputs for the second layer
    A2 = softmax(A1) #calculate the activation of the second layer

    return Z1, A1, Z2, A2

#backward propagation: calculate the error of each layer
def back_prop(Z1, A1, Z2, A2, W2, X, Y):
    one_hot_Y = one_hot(Y)
    dZ2 = A2 - one_hot_Y #calculate the error of the second layer
    dW2 = (1 / m) * dZ2.dot(A1.T) #calculate the gradient of the weights of the second layer
    db2 = (1/ m) * np.sum(dZ2, 1) #calculate the gradient of the biases of the second layer
    dZ1 = W2.T.dot(dZ2) * dReLU(Z1) #calculate the error of the first layer
    dW1 = (1 / m) * dZ1.dot(X.T) #calculate the gradient of the weights of the first layer
    db1 = (1/ m) * np.sum(dZ1, 1) #calculate the gradient of the biases of the first layer
    return dW1, db1, dW2, db2

#update the parameters: subtract the gradients from the weights and biases
#alpha: the learning rate
def update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha):
    W1 -= alpha * dW1 #update the weights of the first layer
    b1 -= alpha * np.reshape(db1, (10, 1)) #update the biases of the first layer
    W2 -= alpha * dW2 #update the weights of the second layer
    b2 -= alpha * np.reshape(db2, (10, 1)) #update the biases of the second layer
    return W1, b1, W2, b2 #return the updated parameters

#get predictions: get the index of the highest value in each row
def get_predictions(A2):
    return np.argmax(A2, 0)

#get accuracy: get the accuracy of the model
def get_accuracy(predictions, Y):
    print(predictions, Y)
    return f"{np.sum(predictions == Y) / Y.size * 100}%"

#gradient descent: we do a little trolling
def gradient_descent(X, Y, iterations, alpha):
    W1, b1, W2, b2 = init_params() #initialize the weights and biases
    for i in range(iterations):
        Z1, A1, Z2, A2 = forward_prop(W1, b1, W2, b2, X)
        dW1, db1, dW2, db2 = back_prop(Z1, A1, Z2, A2, W2, X, Y)
        W1, b1, W2, b2 = update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha)
        if i % 10 == 0:
            print(f"Iteration: {i}")
            print(f"Accuracy: {get_accuracy(get_predictions(A2), Y)}")
    return W1, b1, W2, b2

#main function: run the program
def main():
    W1, b1, W2, b2 = gradient_descent(X_train, Y_train, 250, 0.5)

if __name__ == "__main__":
    main()
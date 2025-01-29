import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
import pandas as pd
from sklearn.svm import LinearSVC

# Workflow: 
# save predictions from model 
# load training and test, fit, predict 
def results_to_csv(y_test, file_name):
    y_test = y_test.astype(int)
    df = pd.DataFrame({'Category': y_test})
    df.index += 1
    df.to_csv(file_name, index_label='Id')


# Load mnist data 
data = np.load("../hw1/data/mnist-data.npz")
mnist_data = data['training_data'] # training data
mnist_data = mnist_data.reshape(mnist_data.shape[0], -1) # (60000, 28*28)
mnist_label = data['training_labels'] # corresponding training labels

mnist_test = data["test_data"]
print("Shape of mnist_test before reshaping:", mnist_test.shape)

mnist_test = mnist_test.reshape(mnist_test.shape[0], -1)
print("Shape of mnist_test after reshaping:", mnist_test.shape)


k_model = LinearSVC(C=0.01)
k_model.fit(mnist_data, mnist_label)

predict = k_model.predict(mnist_test)

results_to_csv(predict, 'MNIST_predictions.csv')



## Load spam data 
s_data = np.load("../hw1/data/spam-data.npz")
spam_data = s_data["training_data"]
spam_data = spam_data.reshape(spam_data.shape[0], -1)

spam_label = s_data["training_labels"]

spam_test = s_data["test_data"]

#create model for predictions 
s_model = LinearSVC(C=0.01)
s_model.fit(spam_data, spam_label)

s_preds = s_model.predict(spam_test)
results_to_csv(s_preds, 'Spam_Predictions.csv')
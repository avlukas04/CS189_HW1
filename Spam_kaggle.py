import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
import pandas as pd
from sklearn.svm import LinearSVC


def results_to_csv(y_test, file_name):
    y_test = y_test.astype(int)
    df = pd.DataFrame({'Category': y_test})
    df.index += 1
    df.to_csv(file_name, index_label='Id')

## Load spam data 
s_data = np.load("../hw1/data/spam-data.npz")
spam_data = s_data["training_data"]
spam_data = spam_data.reshape(spam_data.shape[0], -1)

spam_label = s_data["training_labels"]

spam_test = s_data["test_data"]

#create model for predictions 
s_model = LinearSVC(C=50)
s_model.fit(spam_data, spam_label)

s_preds = s_model.predict(spam_test)
results_to_csv(s_preds, 'Spam_Predictions.csv')

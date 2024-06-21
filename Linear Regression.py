import numpy as np
from sklearn import datasets, linear_model, metrics

# Load diabetes dataset
diabetes = datasets.load_diabetes()
diabetes_X = diabetes.data  # matrix of dimensions 442x10

# Split the data into training/testing sets
diabetes_X_train = diabetes_X[:-20]
diabetes_X_test = diabetes_X[-20:]

# Split the targets into training/testing sets
diabetes_y_train = diabetes.target[:-20]
diabetes_y_test = diabetes.target[-20:]

# with scikit learn:
# Create linear regression object
regr = linear_model.LinearRegression()

# Train the model using the training sets
regr.fit(diabetes_X_train, diabetes_y_train)

# Make predictions using the testing set
diabetes_y_pred = regr.predict(diabetes_X_test)

# The coefficients
print("Coefficients: \n", regr.coef_)
# The mean squared error
mean_squared_error = metrics.mean_squared_error(diabetes_y_test, diabetes_y_pred)
print("Mean squared error: %.2f" % mean_squared_error)
print("="*80)


# train
X = diabetes_X_train
y = diabetes_y_train

# train: init
W = np.zeros(X.shape[1])  # weights (10 parameters given in the task)
b = 0   # bias

learning_rate = 1
epochs = 100000

# train: gradient descent
for i in range(epochs):
    # calculate predictions
    y_pred = np.dot(X, W) + b

    # calculate error and cost (mean squared error)
    mse = metrics.mean_squared_error(y, y_pred) # equation to calculate mse

    # calculate gradients
    grad_w = (1/X.shape[0]) * np.dot(X.T, y_pred-y) # using formula from lecture, x.shape[0]=422, num of patients
    grad_b = (1/X.shape[0]) * np.sum(y_pred-y)

    # update parameters
    W = W - learning_rate * grad_w
    b = b - learning_rate * grad_b

    # diagnostic output
    if i % 5000 == 0:
        print("Epoch %d: %f" % (i, mse))

# coefficients
print("Coefficients: \n", W) # check the coefficients of the training to see if they match the results of scikit-learn
print("="*80)
# test
X_test = diabetes_X_test
y_test = diabetes_y_test

# test: predict
y_pred_test = np.dot(X_test, W) + b # calculate prediction using linear regression

# test: error
mse_test = metrics.mean_squared_error(y_test, y_pred_test)
print("Mean squared error: %.2f" % mse_test)


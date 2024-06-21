import numpy as np
def ID1():
    '''
        Write your personal ID here.
    '''
    # Insert your ID here
    return 316389584
def ID2():
    '''
        Only If you were allowed to work in a pair will you fill this section and place the personal id of your partner otherwise leave it zeros.
    '''
    # Insert your ID here
    return 000000000

def LeastSquares(X,y):
  '''
    Calculates the Least squares solution to the problem
    X*theta=y using the least squares method
    :param X: input matrix
    :param y: input vector
    :return: theta = (Xt*X)^(-1) * Xt * y 
  '''
  theta = np.linalg.inv(X.T @ X) @ X.T @ y
  return theta

def classification_accuracy(model,X,s):
  '''
    calculate the accuracy for the classification problem
    :param model: the classification model class
    :param X: input matrix
    :param s: input ground truth label
    :return: accuracy of the model
  '''
  pred_cl_s = model.predict(X)
  correct_predict= (pred_cl_s==s).sum()
  accuracy = 100*correct_predict/len(s)
  return accuracy

def linear_regression_coeff_submission():
  '''
    copy the values from your notebook into a list in here. make sure the values
    seperated by commas
    :return: list of coefficiants for the linear regression problem.  
  '''
  return [-0.04848220690056652, 0.017705804402269487, -0.024044857442166348, 0.020519977599294484, -0.0068309787871146475, -0.010265139491775706, 0.08427451503883586, -0.0002505941794804365, 0.03554881914118082, -0.010928917674885968, 0.021371344439322818, 0.020220709651072515, 0.07782517783122446, 0.13458932531597687, 0.7994986757385748, 0.024238381642926257, 0.004087009206637526, -0.007541845733499156, 0.006108186503917206, -0.015735599351398847, 0.016912080608902313, -0.0039371658256820916, 0.0008302630663727315, -0.023425739772848193, -0.019370230726838425, -0.022441280029475798, -0.029621861244594042, -0.034473171073741565]

def linear_regression_intrcpt_submission():
  '''
    copy the intercept value from your notebook into here.
    :return: the intercept value.  
  '''
  return 1.1454084680723005e-16

def classification_coeff_submission():
  '''
    copy the values from your notebook into a list in here. make sure the values
    seperated by commas
    :return: list of coefficiants for the classification problem.  
  '''
  return [[-0.06972246741608075, -0.14727727660651382, 0.25552038001848626, -0.29423455424778133, -0.1997694800644474, -0.5286248570362712, -0.025292739041851635, -0.11533439199417708, 0.07361214325848578, 0.5238332369216516, -0.6540403570213895, -0.04715271556101257, -0.07224119058147024, 1.1857995755710775, 2.8124240248200825, -0.4096736889382759, 0.09081960857265756, -0.02455285199580971, -0.2358297585639938, -0.2604766572934806, 0.21163018285452279, 0.0592379414802212, -0.01610113420196579, -0.13431895940486402, -0.4308523898527835, 0.0537520760075251, -0.11850023647908202, 0.41900136181775627]]

def classification_intrcpt_submission():
  '''
    copy the intercept value from your notebook into here.
    :return: the intercept value.  
  '''
  return [0.24970096]

def classification_classes_submission():
  '''
    copy the classes values from your notebook into a list in here. make sure the values
    seperated by commas
    :return: list of classes for the classification problem.  
  '''
  return [-1.0, 1.0]
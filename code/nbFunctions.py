from sklearn.base import BaseEstimator
import numpy as np
import scipy.stats as stats

# For this assignment we will implement the Naive Bayes classifier as a
# a class, sklearn style. You only need to modify the fit and predict functions.
# Additionally, implement the Disparate Impact measure as the evaluateBias function.
class NBC(BaseEstimator):
    '''
    (a,b) - Beta prior parameters for the class random variable
    alpha - Symmetric Dirichlet parameter for the features
    '''

    def __init__(self, a=1, b=1, alpha=1):
        self.a = a
        self.b = b
        self.alpha = alpha
        self.params = None
        
    def get_a(self):
        return self.a

    def get_b(self):
        return self.b

    def get_alpha(self):
        return self.alpha

    # you need to implement this function

    def fit(self,X,y):
        '''
        This function does not return anything
        
        Inputs:
        X: Training data set (N x d numpy array)
        y: Labels (N length numpy array)
        '''
        a = self.get_a()
        b = self.get_b()
        alpha = self.get_alpha()
        self.classes = np.unique(y)

        # remove next line and implement from here
        # you are free to use any data structure for paramse
        params = {}
        total_y = len(y) # total_y = scalar
        count_y_equals_1 = np.count_nonzero(y == 1) # count_y_equals_1 = scalar
        count_y_equals_2 = np.count_nonzero(y == 2) # count_y_equals_2 = scalar
        
        # Equation 5
        theta_bayes = (count_y_equals_1 + a)/(total_y + a + b) # theta_bayes = scalar
        
        # indexes of all 1's
        indices_of_1 = np.where(y == 1)[0] # indices_of_1 = 675,
        
        # indexes of all 2's
        indices_of_2 = np.where(y == 2)[0] # indices_of_2 = 275,
        
        temp_X_1 = X # temp_X_1 = 900 x 18
        temp_X_2 = X # temp_X_2 = 900 x 18
        
        # A matrix of rows corresponding to when y equals 1    
        temp_X_1 = np.delete(temp_X_1, indices_of_2, 0) # temp_X_1 = 625 x 18
            
        # A matrix of rows corresponding to when y equals 2
        temp_X_2 = np.delete(temp_X_2, indices_of_1, 0) # temp_X_2 = 275 x 18
        
        # A vector of the number of unique elements in all the features
        unique_features = [] # unique_features = 0, 
        
        for i in range(np.size(X, 1)):
            unique_features.append(np.unique(X[:,i]))
        
        # unique_features = 18,
        # Sample unique_features = [array([1, 2, 3, 4, 5], dtype=int64), array([1, 2, 3, 4, 5], dtype=int64), array([0, 1, 2, 3, 4, 5], dtype=int64), array([1, 2, 3, 4, 5], dtype=int64), array([1, 2, 3, 4, 5], dtype=int64), array([1, 2, 3, 4, 5, 7], dtype=int64), array([1, 2, 4, 5], dtype=int64), array([1, 2, 3, 4], dtype=int64), array([1, 2, 3, 4, 5], dtype=int64), array([ 1,  2,  3,  5, 10], dtype=int64), array([ 0,  1,  2,  3,  4,  5, 10], dtype=int64), array([1, 2, 3, 4, 8], dtype=int64), array([1, 2, 3, 4], dtype=int64), array([1, 2, 3, 4, 5], dtype=int64), array([1, 2, 3, 5], dtype=int64), array([1, 2, 3, 4, 5], dtype=int64), array([1, 2, 3, 4, 5], dtype=int64), array([1, 2, 3, 4, 5], dtype=int64)]
        # Basically an array of arrays listing unique features in each column

        # Equation 8 
        
        # So theta_1_j will have a dictionary containing the feature xj and it's probability when Y = 1
        # We only need to foxus on temp_X_1 for equation 8
        theta_1_j = {}
        
        # outer loop to access the columns and its respective unique features
        for i in range(np.size(temp_X_1, 1)):
            
            # inner loop to compute probability for each feature which will be stored in a dictionary
            for j in unique_features[i]:
                count_j = np.count_nonzero(temp_X_1[:,i] == j) # count_j = scalar
                probability = (count_j + alpha)/(count_y_equals_1 + (len(unique_features[i]) * alpha)) # probability = scalar
                
                # Add the j and its probability to theta_1_j in the inner loop
                temp_name  = "c" + str(i) + "f" + str(j) # temp_name = string
                theta_1_j[temp_name] = probability
                
            # Default probability if key doesn't exist
            probability = (0 + alpha)/(count_y_equals_1 + (len(unique_features[i]) * alpha)) # probability = scalar
            
            # Default name 
            temp_name  = "c" + str(i) + "fd" # temp_name = string
            theta_1_j[temp_name] = probability
            
        # Equation 9
        
        # So theta_2_j will have a dictionary containing the feature xj and it's probability when Y = 2
        # We only need to foxus on temp_X_2 for equation 9
        theta_2_j = {}
        
        # outer loop to access the columns and its respective unique features
        for i in range(np.size(temp_X_2, 1)):

            # inner loop to compute probability for each feature which will be stored in a dictionary
            for j in unique_features[i]:
                count_j = np.count_nonzero(temp_X_2[:,i] == j) # count_j = scalar
                probability = (count_j + alpha)/(count_y_equals_2 + (len(unique_features[i]) * alpha)) # probability = scalar
                
                # Add the j and its probability to theta_2_j in the inner loop
                temp_name  = "c" + str(i) + "f" + str(j) # temp_name = string
                theta_2_j[temp_name] = probability
                
            # Default probability if key doesn't exist
            probability = (0 + alpha)/(count_y_equals_2 + (len(unique_features[i]) * alpha)) # probability = scalar
            
            # Default name
            temp_name  = "c" + str(i) + "fd" # temp_name = string
            theta_2_j[temp_name] = probability
        
        # Add theta_bayes, theta_1_j and theta_2_j to the dictionary params
        params['theta_bayes'] = theta_bayes
        params['theta_1_j'] = theta_1_j
        params['theta_2_j'] = theta_2_j

        # do not change the line below
        self.params = params
    
    # you need to implement this function
    def predict(self,Xtest):
        '''
        This function returns the predicted class for a given data set
        
        Inputs:
        Xtest: Testing data set (N x d numpy array)
        
        Output:
        predictions: N length numpy array containing the predictions
        '''
        params = self.params
        a = self.get_a()
        b = self.get_b()
        alpha = self.get_alpha()
        #remove next line and implement from here
        
        # Accessing the dictionary params to get eq5, eq8 and eq9 
        p_y_1 = params['theta_bayes'] # p_y_1 = scalar
        p_y_2 = 1 - p_y_1 # p_y_2 = scalar
        
        theta_1_j = params['theta_1_j']
        theta_2_j = params['theta_2_j']
        
        predictions = []
        
        # Outer loop to loop through columns of XTest
        for i in range(np.size(Xtest, 0)):
            product_p_x_xj_y_1 = 1 # product_p_x_xj_y_1 = scalar
            product_p_x_xj_y_2 = 1 # product_p_x_xj_y_2 = scalar
            
            temp_row = Xtest[i] # Extracting row easily
            
            # Inner loop to loop through each feature and multiply the probability
            for j in range(np.size(temp_row)):
                
                # Need to compute for both cases when Y == 1 and Y == 2 to compare
                
                # Name
                temp_name = "c" + str(j) + "f" + str(temp_row[j]) # temp_name = string
                # If the key exists
                if temp_name in theta_1_j.keys():
                    product_p_x_xj_y_1 = product_p_x_xj_y_1 * theta_1_j[temp_name] # product_p_x_xj_y_1 = scalar
                else:
                # If the key doesn't exist
                    # Default Name
                    temp_name  = "c" + str(j) + "fd" # temp_name = string
                    product_p_x_xj_y_1 = product_p_x_xj_y_1 * theta_1_j[temp_name] # product_p_x_xj_y_1 = scalar
                
                # Name
                temp_name = "c" + str(j) + "f" + str(temp_row[j]) # temp_name = string
                # If the key exists
                if temp_name in theta_2_j.keys():
                    product_p_x_xj_y_2 = product_p_x_xj_y_2 * theta_2_j[temp_name] # product_p_x_xj_y_2 = scalar
                else:
                # If the key doesn't exist
                    # Default Name
                    temp_name  = "c" + str(j) + "fd" # temp_name = string
                    product_p_x_xj_y_2 = product_p_x_xj_y_2 * theta_2_j[temp_name] # product_p_x_xj_y_2 = scalar
            
            # Equation 3:
            p_y_1_X_x = (p_y_1 * product_p_x_xj_y_1)/((p_y_1 * product_p_x_xj_y_1) + (p_y_2 * product_p_x_xj_y_2)) # p_y_1_X_x = scalar         
            
            # Equation 4:
            p_y_2_X_x = (p_y_2 * product_p_x_xj_y_2)/((p_y_1 * product_p_x_xj_y_1) + (p_y_2 * product_p_x_xj_y_2)) # p_y_2_X_x = scalar

            # Compare which is better and assign labels accordingly
            if (p_y_1_X_x > p_y_2_X_x):
                predictions.append(1)
            else:
                predictions.append(2)

        # predictions = np.random.choice(self.classes,np.unique(Xtest.shape[0]))
        # do not change the line below
        # print(predictions)
        return predictions
        
def evaluateBias(y_pred,y_sensitive):
    '''
    This function computes the Disparate Impact in the classification predictions (y_pred),
    with respect to a sensitive feature (y_sensitive).
    
    Inputs:
    y_pred: N length numpy array
    y_sensitive: N length numpy array
    
    Output:
    di (disparateimpact): scalar value
    '''
    #remove next line and implement from here
    di = 0
    
    # Count of all 1's in y_sensitive
    count_y_equals_1 = np.count_nonzero(y_sensitive == 1) # count_y_equals_1 = scalar
    
    # Count of all 2's in y_sensitive
    count_y_equals_2 = np.count_nonzero(y_sensitive == 2) # count_y_equals_2 = scalar
        
    # indexes of all 1's of y_sensitive
    indices_of_1 = np.where(y_sensitive == 1) # indices_of_1 = 690,
        
    # indexes of all 2's of y_sensitive
    indices_of_2 = np.where(y_sensitive == 2) # indices_of_2 = 310,
    
    temp_y_1 = y_pred # temp_X_1 = 900 x 18
    temp_y_2 = y_pred # temp_X_2 = 900 x 18
        
    # A list of rows corresponding to when y_sensitive equals 1    
    temp_y_1 = np.delete(temp_y_1, indices_of_2) # temp_y_1 = 690,
            
    # A list of rows corresponding to when y_sensitive equals 2
    temp_y_2 = np.delete(temp_y_2, indices_of_1) # temp_y_2 = 310,
    
    # Count of all the 2's in y_pred when y_sensitive is 2
    temp_y_2_s_2 = np.count_nonzero(temp_y_2 == 2) # temp_y_2_s_2 = scalar
    
    # Count of all the 2's in y_pred when y_sensitive is 1
    temp_y_2_s_1 = np.count_nonzero(temp_y_1 == 2) # temp_y_2_s_1 = scalar
    
    # Numerator for Disparate Impact
    y_2_s_2 = temp_y_2_s_2/count_y_equals_2 # y_2_s_2 = scalar
    
    # Denominator for Disparate Impact
    y_2_s_1 = temp_y_2_s_1/count_y_equals_1 # y_2_s_1 = scalar
    
    # Disparate Impact
    di = y_2_s_2/y_2_s_1 # di = scalar

    #do not change the line below
    return di

def genBiasedSample(X,y,s,p,nsamples=1000):
    '''
    Oversamples instances belonging to the sensitive feature value (s != 1)
    
    Inputs:
    X - Data
    y - labels
    s - sensitive attribute
    p - probability of sampling unprivileged customer
    nsamples - size of the resulting data set (2*nsamples)
    
    Output:
    X_sample,y_sample,s_sample
    '''
    i1 = y == 1 # good
    i1 = i1[:,np.newaxis]
    i2 = y == 2 # bad
    i2 = i2[:,np.newaxis]
    
    sp = s == 1 #privileged
    sp = sp[:,np.newaxis]
    su = s != 1 #unprivileged
    su = su[:,np.newaxis]

    su1 = np.where(np.all(np.hstack([su,i1]),axis=1))[0]
    su2 = np.where(np.all(np.hstack([su,i2]),axis=1))[0]
    sp1 = np.where(np.all(np.hstack([sp,i1]),axis=1))[0]
    sp2 = np.where(np.all(np.hstack([sp,i2]),axis=1))[0]
    inds = []
    for i in range(nsamples):
        u = stats.bernoulli(p).rvs(1)
        if u == 1:
            #sample one bad instance with s != 1
            inds.append(np.random.choice(su2,1)[0])
            #sample one good instance with s == 1
            inds.append(np.random.choice(sp1,1)[0])
        else:
            #sample one good instance with s != 1
            inds.append(np.random.choice(su1,1)[0])
            #sample one bad instance with s == 1
            inds.append(np.random.choice(sp2,1)[0])
    X_sample = X[inds,:]
    y_sample = y[inds]
    s_sample = s[inds]
    
    return X_sample,y_sample,s_sample,inds
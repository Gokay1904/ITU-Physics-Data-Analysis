import numpy as np
class GaussianNB:
    def __init__(self, priors=None, var_smoothing=1e-9):
        self.priors = priors
        self.var_smoothing = var_smoothing

    def logprior(self, class_ind):
   
        return np.log(self.class_priors_[class_ind] + self.var_smoothing)

    def loglikelihood(self, Xi, class_ind):

        mu = self.theta_[class_ind]
        var = self.var_[class_ind]
        
        GaussLikelihood = (1 / (var * np.sqrt(2 * np.pi))) * np.exp(-1 * (Xi - mu)**2 / (2 * (var**2)) )
     
        logGaussLikelihood = np.log(GaussLikelihood + self.var_smoothing)

      
        return logGaussLikelihood

    def posterior(self, Xi, class_ind):
        logprior = self.logprior(class_ind)
        loglikelihood = self.loglikelihood(Xi, class_ind)
 
        return logprior + np.sum(loglikelihood)

    def fit(self, X, y):
     
        n_samples, n_features = X.shape
        
        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)
        
        self.theta_ = np.zeros((n_classes,n_features)) 
        self.var_ =  np.zeros((n_classes,n_features)) 
     
        self.class_priors_ = np.zeros(n_classes)
 
        for c_ind, c_id in enumerate(self.classes_):
  
            X_class = X[y == c_id]     
            
            self.theta_[c_ind,:] = np.mean(X_class, axis=0)
            self.var_[c_ind,:] = np.var(X_class)
            self.class_priors_[c_ind] = X_class.shape[0] / n_samples
        
    def predict(self, X):
        y_pred = []
        for Xi in X.values:
            posteriors = []   
            for class_ind in self.classes_:          
                sample_posterior = self.posterior(Xi, class_ind)
                posteriors.append(sample_posterior) 
            y_pred.append(np.argmax(posteriors))
        
        return y_pred
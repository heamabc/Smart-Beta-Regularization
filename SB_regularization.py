from scipy.optimize import minimize
import numpy as np

# ==================================== Super class model ==============================================
class super_model:
    def __init__(self, return_df, weight=None, lambda_=1, regularization=False):
        self.lambda_ = lambda_
        self.regularization = regularization

        self.return_df = return_df

    @staticmethod
    def L2_regularization(w):
        return np.sum(w**2)

    def loss_function_with_regularization(self, w, cov):
        return self.loss_function(w, cov) + self.lambda_ * self.L2_regularization(w)

    def fit(self, long=True):
        cov = self.return_df.cov()
        n = cov.shape[0]
        weights = np.ones(n)/n
        cons = ({'type': 'eq', 'fun': lambda x:1-sum(x)})
        bnds = [(0,0.1) for i in weights]
        
        if long and self.regularization:
            res = minimize(self.loss_function_with_regularization, weights, args=(cov), method='SLSQP', constraints=cons, bounds = bnds)
        elif long and not self.regularization:
            res = minimize(self.loss_function, weights, args=(cov), method='SLSQP', constraints=cons, bounds = bnds)
        elif not long and self.regularization:
            res = minimize(self.loss_function_with_regularization, weights, args=(cov), method='SLSQP', constraints=cons)
        elif not long and not self.regularization:
            res = minimize(self.loss_function, weights, args=(cov), method='SLSQP', constraints=cons)

        self.weight = res.x
        return

    def calc_return(self, ):
        daily_return = (self.weight * self.return_df).sum(axis=1)
        culmulative_return = (daily_return + 1).cumprod()

        return daily_return, culmulative_return

# ==================================== MDR model subclass ==============================================
class MDR_model(super_model):

    @staticmethod
    def loss_function(w, cov):
        w_vol = np.sqrt(np.diag(cov)).dot(w.T)
        port_vol = np.sqrt(w.dot(cov).dot(w))
        diversification_ratio = w_vol/port_vol
        return -diversification_ratio

# ==================================== MSR model subclass ==============================================
class MSR_model(super_model):
    
    def loss_function(self, w, cov):
        mu = w.dot(self.return_df.mean())
        volatility = np.sqrt(w.dot(cov).dot(w))
        return -mu/volatility

# ==================================== GMV model subclass ==============================================
class GMV_model(super_model):

    @staticmethod
    def loss_function(w, cov):
        return w.dot(cov).dot(w)

# ==================================== Equal weight model subclass ==============================================
class equal_weight_model(super_model):

    def __init__(self, return_df, weight=None, lambda_=1, regularization=False):
        super_model.__init__(self, return_df, weight, lambda_, regularization)
        self.weight = np.ones(return_df.shape[1]) * (1 / (return_df.shape[1]))
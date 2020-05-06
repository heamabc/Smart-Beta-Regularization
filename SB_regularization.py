from scipy.optimize import minimize
import numpy as np
import pandas as pd
from tqdm import tqdm

# ==================================== Super class model ==============================================
class super_model:
    def __init__(self, return_df, weight=None, lambda_=1, regularization=False, rebalance=None):
        self.lambda_ = lambda_
        self.regularization = regularization
        self.rebalance = rebalance

        self.return_df = return_df
        self.weight =[]

    @staticmethod
    def L2_regularization(w):
        return np.sum(w**2)

    def loss_function_with_regularization(self, w, cov):
        return self.loss_function(w, cov) + self.lambda_ * self.L2_regularization(w)
    
    def fit(self):
        if self.rebalance==None:
            self.model(self.return_df)
        elif self.rebalance=="Q":
            
            quarterly_periods = self.return_df.index.to_period('Q').unique()
            for this_quarter in tqdm(quarterly_periods, desc=self.name):
                this_quarter_df = self.return_df.loc[self.return_df.index.to_period('Q') == this_quarter]
                self.model(this_quarter_df)
        print('Done!')
        return self
        
    def model(self, return_df, long=True):
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
        
        self.weight.append(res.x)
        return

    def calc_return(self, ):
        if self.rebalance == None:
          daily_return = (self.weight[0] * self.return_df).sum(axis=1)
          culmulative_return = (daily_return + 1).cumprod()
        elif self.rebalance == "Q":
          daily_return = pd.Series()

          quarterly_periods = self.return_df.index.to_period('Q').unique()
          for this_quarter in range(1,len(quarterly_periods)):
            this_quarter_return = self.return_df.loc[self.return_df.index.to_period('Q') == quarterly_periods[this_quarter]]
            this_quarter_daily_rtn = (this_quarter_return * self.weight[this_quarter]).sum(axis=1)
            this_quarter_culmulative_rtn = (this_quarter_daily_rtn + 1).cumprod()

            daily_return = pd.concat([daily_return, this_quarter_daily_rtn])

          culmulative_return = (daily_return + 1).cumprod()

        return daily_return, culmulative_return

# ==================================== MDR model subclass ==============================================
class MDR_model(super_model):

    def __init__(self, return_df, weight=None, lambda_=1, regularization=False, rebalance=None):
        super_model.__init__(self, return_df, weight, lambda_, regularization, rebalance)
        self.name = "MDR"

    @staticmethod
    def loss_function(w, cov):
        w_vol = np.sqrt(np.diag(cov)).dot(w.T)
        port_vol = np.sqrt(w.dot(cov).dot(w))
        diversification_ratio = w_vol/port_vol
        return -diversification_ratio

# ==================================== MSR model subclass ==============================================
class MSR_model(super_model):

    def __init__(self, return_df, weight=None, lambda_=1, regularization=False, rebalance=None):
        super_model.__init__(self, return_df, weight, lambda_, regularization, rebalance)
        self.name = "MSR"

    def loss_function(self, w, cov):
        mu = w.dot(self.return_df.mean())
        volatility = np.sqrt(w.dot(cov).dot(w))
        return -mu/volatility

# ==================================== GMV model subclass ==============================================
class GMV_model(super_model):

    def __init__(self, return_df, weight=None, lambda_=1, regularization=False, rebalance=None):
        super_model.__init__(self, return_df, weight, lambda_, regularization, rebalance)
        self.name = "GMV"

    @staticmethod
    def loss_function(w, cov):
        return w.dot(cov).dot(w)

# ==================================== Equal weight model subclass ==============================================
class equal_weight_model(super_model):

    def __init__(self, return_df, weight=None, lambda_=1, regularization=False, rebalance=None):
        super_model.__init__(self, return_df, weight, lambda_, regularization, rebalance)
        self.weight = tuple(np.ones(return_df.shape[1]) * (1 / (return_df.shape[1])))
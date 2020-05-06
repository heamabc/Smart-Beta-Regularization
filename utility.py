import pandas as pd
import numpy as np
import re

class data_generation:
    
    def __init__(self, input_directory, start_date, end_date):
        self.input_directory = input_directory
        self.start_date = start_date
        self.end_date = end_date
        
    @staticmethod
    def dropna_cols(df):
        return df.iloc[1:].dropna(axis=1)

    @staticmethod
    def drop_SPY(df):
        return df.drop(columns=['SPY'])

    @staticmethod
    def transform_ln_price(open_df):

        # If the first valid index is none, return first element, else return none
        def first_valid_index(series):
          first_index = series.first_valid_index()
          if first_index != None:
            return series.loc[first_index]
          else:
            return None

        ln_open_df = np.log(open_df/open_df.apply(first_valid_index))

        return ln_open_df
        
    def output_data(self, dropna=False, dropSPY=False):
        
        # Read data and slicing
        data = pd.read_csv(self.input_directory, index_col=0)
        data = data.loc[self.start_date:self.end_date]
        
        # Tickers
        regex_pat = re.compile(r'_.*')
        Tickers = data.columns.str.replace(r'_.*','').unique()[1:]

        # Calculate effect of split
        splitFactor_cols = [ele + '_splitFactor' for ele in Tickers]
        splitFactor_df = data[splitFactor_cols]
        splitFactor_df = splitFactor_df.cumprod()
        splitFactor_df.columns = Tickers

        # Calculate effect of split cash dividend
        divCash_cols = [ele + '_divCash' for ele in Tickers]
        divCash_df = data[divCash_cols]
        divCash_df = divCash_df.cumsum()
        divCash_df.columns = Tickers

        # Slice open, close, volumne df
        open_cols = [ele + '_open' for ele in Tickers]
        close_cols = [ele + '_close' for ele in Tickers]
        volume_cols = [ele + '_volume' for ele in Tickers]

        open_df = data[open_cols]
        close_df = data[close_cols]
        volume_df = data[volume_cols]

        open_df.columns = Tickers
        close_df.columns = Tickers
        volume_df.columns = Tickers

        open_df = open_df * splitFactor_df + divCash_df * splitFactor_df
        close_df = close_df * splitFactor_df + divCash_df * splitFactor_df
        volume_df = volume_df * splitFactor_df

        # Return
        open_return_df = open_df.pct_change()
        close_return_df = close_df.pct_change()
        
        ln_open_df = self.transform_ln_price(open_df)

        if dropna == True:
            ln_open_df, open_df, close_df, volume_df, open_return_df, close_return_df = map( self.dropna_cols, [ln_open_df, open_df, close_df, volume_df, open_return_df, close_return_df])

        if dropSPY == True:
            ln_open_df, open_df, close_df, volume_df, open_return_df, close_return_df = map( self.drop_SPY, [ln_open_df, open_df, close_df, volume_df, open_return_df, close_return_df])

        # Some columns might be dropped
        Tickers = list(ln_open_df.columns)

        ln_open_np = ln_open_df.values
        
        SPY_culmulative_return = (open_return_df['SPY'].iloc[1:] + 1).cumprod()

        return ln_open_np, open_df, close_df, volume_df, open_return_df, close_return_df, Tickers, SPY_culmulative_return


class performance_generation:

    @staticmethod
    def sharpe(port_daily_return):
        return np.sqrt(252) * (port_daily_return.mean() / port_daily_return.std())

    @staticmethod
    def drawdown(port_culmulative_return):

        expanding_max = port_culmulative_return.expanding().max()
        expanding_min = port_culmulative_return.expanding().min()

        drawdown = expanding_min/expanding_max - 1
        max_drawdown = drawdown.iloc[-1]

        return drawdown, max_drawdown

    @staticmethod
    def annualized_return(port_culmulative_return):
        last_date = datetime.datetime.strptime(port_culmulative_return.index[-1], '%m/%d/%Y')
        first_date = datetime.datetime.strptime(port_culmulative_return.index[0], '%m/%d/%Y')
        days = (last_date - first_date).days

        return port_culmulative_return.iloc[-1]**(365/days) - 1

    @staticmethod
    def annualized_volatility(port_daily_return):
        return port_culmulative_return.std() * np.sqrt(252)

    def main(self, port_daily_return, port_culmulative_return):
        sharpe = self.sharpe(port_daily_return)
        drawdown, max_drawdown = self.drawdown(port_culmulative_return)
        annualized_return = self.annualized_return(port_culmulative_return)
        annualized_volatility = self.annualized_volatility(port_daily_return)

        return sharpe, drawdown, max_drawdown, annualized_return, annualized_volatility
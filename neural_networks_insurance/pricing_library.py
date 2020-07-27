import pandas as pd
import numpy as np
   

class MarketEvaluationTools():
    """
    PLACEHOLDER DOCSTRING:
        takes in a model dict, a prices df and a vector of claims
        has functionality to compute a series of metrics
        naming convention for methods is important as it is used later
    """

    def __init__(self, model_names, prices_df, claims):
        self.model_names = model_names
        self.priceMat = prices_df
        self.claims = claims
        self.contracts_per_firm = self._get_contracts_won_per_firm()
        self.price_offered_per_firm = self._get_prices_offered_per_firm()
        self.prices_market = self._get_prices_won()

    def _get_contracts_won_per_firm(self):

        pdf = self.priceMat
        new_order = list(pdf.columns)
        np.random.shuffle(new_order)  # pick randomly among equal prices
        winners = pdf[new_order].idxmin(axis=1)
        contracts_per_firm = {}
        for key in self.model_names:
            contracts = winners[winners == key].index
            contracts = list(contracts)
            if len(contracts) != 0:
                contracts_per_firm[key] = contracts
        return contracts_per_firm

    def _get_prices_won(self):
        return self.priceMat.min(axis=1)

    def _get_prices_offered_per_firm(self):
        price_dict = {}
        for key in self.model_names:
            if key in self.contracts_per_firm:
                idx = self.contracts_per_firm[key]
                price_dict[key] = self.priceMat[key].loc[idx]
            else:
                price_dict[key] = []
        return price_dict

    def get_loss_per_firm(self):
        costs = {}
        for key in self.model_names:
            if key in self.contracts_per_firm:
                idx = self.contracts_per_firm[key]
                mod_cost = self.claims.loc[idx].sum()
            else:
                mod_cost = 0
            costs[key] = mod_cost
        return pd.Series(costs)

    def get_loss_per_contract_per_firm(self):
        losses = self.get_loss_per_firm()
        mean_loss = {}
        for key in self.model_names:
            if key in self.contracts_per_firm:
                nconts = len(self.contracts_per_firm[key])
                mean_loss[key] = losses.loc[key] / nconts
            else:
                mean_loss[key] = 0
        return pd.Series(mean_loss)

    def get_revenue_per_firm(self):

        revs = {}
        for key in self.model_names:
            if key in self.contracts_per_firm:
                idx = self.contracts_per_firm[key]
                mod_rev = self.priceMat[key].loc[idx].sum()
            else:
                mod_rev = 0
            revs[key] = mod_rev
        return pd.Series(revs)

    def get_profit_per_firm(self):
        revs = self.get_revenue_per_firm()
        losses = self.get_loss_per_firm()
        profs = revs - losses
        return profs

    def get_mean_price_offered_per_firm(self):
        return self.priceMat.mean(axis=0)

    def get_mean_price_won_per_firm(self):
        means = {}
        for key in self.model_names:
            if key in self.contracts_per_firm:
                idx = self.contracts_per_firm[key]
                new_prices = self.priceMat[key].loc[idx]
                means[key] = new_prices.mean(axis=0)
            else:
                means[key] = 0
        return pd.Series(means)

    def get_market_shares(self):
        mshares = {}
        msize = 0
        for key in self.model_names:
            if key in self.contracts_per_firm:
                share = len(self.contracts_per_firm[key])
            else:
                share = 0
            mshares[key] = share
            msize += share

        mshares = pd.Series(mshares)
        mshares = mshares / msize
        return mshares

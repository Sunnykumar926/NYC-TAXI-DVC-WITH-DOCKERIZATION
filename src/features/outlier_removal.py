from sklearn.base import BaseEstimator, TransformerMixin, OneToOneFeatureMixin
import pandas as pd
import numpy as np

class OutliersRemover(BaseEstimator, TransformerMixin, OneToOneFeatureMixin):
    def __init__(self, percentile_value: list, col_subset:list):
        self.percentile_value = percentile_value
        self.col_subset = col_subset

    def fit(self, x, y=None):
        x = x.copy()
        self.quantiles = []

        for col in self.col_subset:
            lower_bound = x[col].quantile(q=self.percentile_value[0])
            upper_bound = x[col].quantile(q=self.percentile_value[1])

            self.quantiles.append((lower_bound, upper_bound))
        
        return self
    
    def transform(self, x):
        x = x.copy()

        for ind, col in enumerate(self.col_subset):
            lower_bound, upper_bound = self.quantiles[ind]
            filter_df = x[(x[col]>=lower_bound) & (x[col]<=upper_bound)]
            x = filter_df
        return x

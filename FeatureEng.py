from sklearn.base import BaseEstimator, TransformerMixin


class FeatureEngineering(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return self.feature_eng(X)

    @staticmethod
    def feature_eng(df):
        # engineering the numerical features
        # age, gender, satisfaction interaction
        df['age_gender_satisfaction'] = df['CustomerAge'] + df['CustomerGender'] + df['CustomerSatisfaction']

        # age and gender interaction
        df['age_gender'] = df['CustomerAge'] * df['CustomerGender']

        # age and customerSatisfaction interaction
        df['age_customerSatisfaction'] = df['CustomerAge'] * df['CustomerSatisfaction']

        # gender and customerSatisfaction interaction
        df['gender_customerSatisfaction'] = df['CustomerGender'] * df['CustomerSatisfaction']

        # age and Productprice interaction
        df['productprice_age'] = df['ProductPrice'] * df['CustomerAge']

        # purchasefrequency and gender interaction
        df['purchasefrequency_gender'] = df['CustomerGender'] * df['PurchaseFrequency']

        # customersatisfaction and Purchase frequency
        df['custsat_purfreq'] = df['CustomerSatisfaction'] * df['PurchaseFrequency']

        # producprice and gender interaction
        df['productprice_gender'] = df['ProductPrice'] * df['CustomerGender']

        return df
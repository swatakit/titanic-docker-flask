import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from utils.CustomTransformers import ColumnExtractor, DFImputer, DFStandardScaler, DummyTransformer, DFFeatureUnion, PredictProba
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform

def hypertune_titanic_model():
    """Performs hyperparameter tuning for a logistic regression model using random grid search."""

    # Fetch the dataset
    titanic = pd.read_csv('./data/titanic_train.csv')
    y = titanic['Survived']
    X = titanic.drop(columns=['Survived'])

    feat_cats = ['Sex', 'Embarked']
    feat_nums = ['Age']
    feat_donth = ['Pclass', 'SibSp']
    pipeline_titanic = Pipeline([
        ('features', DFFeatureUnion([
            ('numerics', Pipeline([
                ('extract', ColumnExtractor(feat_nums)),
                ('mid_fill', DFImputer(strategy='median')),
                ('scale', DFStandardScaler())
            ])),
            ('categoricals', Pipeline([
                ('extract', ColumnExtractor(feat_cats)),
                ('dummy', DummyTransformer())
            ])),
            ('raw', Pipeline([
                ('extract', ColumnExtractor(feat_donth)),
            ]))
        ]))
    ])

    # Create a logistic regression model
    logit = LogisticRegression(random_state=42, class_weight='balanced', n_jobs=-1)

    # Create a pipeline with the data and the model
    pipeline = Pipeline([('data', pipeline_titanic), ('logit_hyper',logit)])

    # Define the hyperparameters to tune
    param_dist = {
        'logit_hyper__C': uniform(0.1, 10),
        'logit_hyper__penalty': ['l1', 'l2']
    }

    # Perform randomized search to find the best hyperparameters
    logit_search = RandomizedSearchCV(pipeline, param_distributions=param_dist, cv=5, n_jobs=-1, n_iter=20, random_state=42)
    logit_search.fit(X, y)

    # Print the best hyperparameters
    print("Best hyperparameters:")
    print(logit_search.best_params_)

    # Save the best model
    joblib.dump(logit_search, 'titanic_model.sav')
    print(f"Model saved to: titanic_model.sav")



if __name__ == '__main__':
    # Tune the hyperparameters
    hypertune_titanic_model()
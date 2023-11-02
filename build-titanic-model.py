import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from utils.CustomTransformers import ColumnExtractor, DFImputer, DFStandardScaler, DummyTransformer, DFFeatureUnion, PredictProba
from sklearn.metrics import classification_report, confusion_matrix

def evaluate_model(model, X_test, y_test):
    """Evaluates a model on the test data and prints the accuracy, confusion matrix, and classification report."""
    y_pred = model.predict(X_test)
    accuracy = model.score(X_test, y_test)
    print(f"Model accuracy on test data: {accuracy:.2f}")

    # Print the confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:")
    print(cm)

    # Print the classification report
    cr = classification_report(y_test, y_pred)
    print("Classification Report:")
    print(cr)

def build_titanic_model():
    """Builds and saves a logistic regression model for the Titanic dataset."""
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

    # Fetch the dataset
    titanic = pd.read_csv('./data/titanic_train.csv')
    y = titanic['Survived']
    X = titanic.drop(columns=['Survived'])

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create a logistic regression, fit and save model
    logit = LogisticRegression(random_state=42, class_weight='balanced', n_jobs=-1)
    logit.fit(pipeline_titanic.fit_transform(X_train), y_train)

    # Evaluate the model
    evaluate_model(logit, pipeline_titanic.transform(X_test), y_test)


    # Save the model
    model = Pipeline([('data', pipeline_titanic), ('logit', PredictProba(logit))])
    joblib.dump(model, 'titanic_model.sav')
    print(f"Model saved to: titanic_model.sav")


if __name__ == '__main__':
    build_titanic_model()
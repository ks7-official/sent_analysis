import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, mean_squared_error
import joblib
import os

def run_models(data_path='data/merged_data.csv'):
    if not os.path.exists(data_path):
        print(f"❌ File not found: {data_path}")
        return

    df = pd.read_csv(data_path).dropna()

    df['label'] = df['classification'].map({'Fear': 0, 'Neutral': 1, 'Greed': 2})
    X_cls = df[['sentiment_score', 'Leverage']]
    y_cls = df['label']

    X_reg = df[['sentiment_score', 'Leverage']]
    y_reg = df['Closed PnL']

    Xc_train, Xc_test, yc_train, yc_test = train_test_split(X_cls, y_cls, test_size=0.2, random_state=42)
    Xr_train, Xr_test, yr_train, yr_test = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)

    clf = RandomForestClassifier().fit(Xc_train, yc_train)
    acc = accuracy_score(yc_test, clf.predict(Xc_test))

    reg = RandomForestRegressor().fit(Xr_train, yr_train)
    mse = mean_squared_error(yr_test, reg.predict(Xr_test))

    os.makedirs("models", exist_ok=True)
    joblib.dump(clf, 'models/sentiment_classifier.pkl')
    joblib.dump(reg, 'models/pnl_regressor.pkl')

    print(f"✅ Classification Accuracy: {acc:.2f}")
    print(f"✅ Regression MSE: {mse:.2f}")
    print("✅ Models saved to 'models/'")

if __name__ == "__main__":
    run_models()

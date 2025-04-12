import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from joblib import dump

# Generate synthetic training data
def generate_training_data(num_samples=500):
    data = []
    for _ in range(num_samples):
        price = np.random.uniform(10, 500)
        clicks = np.random.randint(1, 100)
        sales = price * 0.3 + clicks * 2 + np.random.normal(0, 10)
        data.append({'price': price, 'clicks': clicks, 'sales': sales})
    return pd.DataFrame(data)

if __name__ == "__main__":
    df = generate_training_data()
    X = df[['price', 'clicks']]
    y = df['sales']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    dump(model, "model.pkl")
    print("ML model trained and saved as ml_model/model.pkl")

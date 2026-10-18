
import pandas as pd
import joblib
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor

# Load dataset
data = pd.read_csv("datasets/car details v3.csv")

# Select features and target
features = [
    "name", "year", "km_driven", "fuel",
    "seller_type", "transmission", "owner",
    "mileage", "engine", "max_power", "seats"
]

target = "selling_price"

data = data[features + [target]].copy()

# Convert numeric columns
numeric_columns = [
    "year", "km_driven", "mileage",
    "engine", "max_power", "seats"
]

for col in numeric_columns + [target]:
    data[col] = pd.to_numeric(
        data[col].astype(str).str.extract(r"([-+]?\d*\.?\d+)")[0],
        errors="coerce"
    )

# Remove rows without a price
data = data.dropna(subset=[target])

X = data[features]
y = data[target]

categorical_columns = [
    "name", "fuel", "seller_type", "transmission", "owner"
]

preprocessor = ColumnTransformer([
    ("numeric", SimpleImputer(strategy="median"), numeric_columns),
    ("categorical", Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ]), categorical_columns)
])

model = Pipeline([
    ("preprocessor", preprocessor),
    ("regressor", RandomForestRegressor(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    ))
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model.fit(X_train, y_train)

print("Model trained successfully!")

y_pred = model.predict(X_test)

r2 = model.score(X_test, y_test)
mae = mean_absolute_error(y_test, y_pred)

print("R2 Score:", round(r2, 3))
print("MAE:", round(mae, 2))

joblib.dump(model, "car_price_model.pkl")
print("Model saved successfully!")

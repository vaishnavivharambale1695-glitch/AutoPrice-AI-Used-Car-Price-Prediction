
import streamlit as st
import pandas as pd
import joblib
import os
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# ---------------- PAGE CONFIGURATION ----------------
st.set_page_config(
    page_title="AutoPrice AI",
    page_icon="🚗",
    layout="wide"
)

# ---------------- LOAD MODEL AND DATA ----------------
@st.cache_resource
def load_model():
    return joblib.load("car_price_model.pkl")

@st.cache_data
def load_data():
    df = pd.read_csv("datasets/car details v3.csv")

    numeric_columns = [
        "year", "km_driven", "mileage",
        "engine", "max_power", "seats", "selling_price"
    ]

    for col in numeric_columns:
        df[col] = pd.to_numeric(
            df[col].astype(str).str.extract(
                r"([-+]?\d*\.?\d+)"
            )[0],
            errors="coerce"
        )

    return df

try:
    model = load_model()
    data = load_data()
except Exception as e:
    st.error(f"Could not load model or dataset: {e}")
    st.stop()

history_file = "prediction_history.csv"

if os.path.exists(history_file):
    history = pd.read_csv(history_file)
else:
    history = pd.DataFrame(
        columns=["Car Model", "Predicted Price", "Date"]
    )

# ---------------- SIDEBAR ----------------
st.sidebar.markdown(
    "# 🚗 AutoPrice AI\nSmart Used Car Price Prediction"
)
st.sidebar.divider()

page = st.sidebar.radio(
    "Navigate",
    [
        "Dashboard",
        "Price Prediction",
        "Analytics",
        "Prediction History",
        "Car Price Comparison",
        "Model Performance"
    ]
)

# ---------------- STYLING ----------------
st.markdown("""
<style>
.stApp {
    background-color: #f3f6fb;
}
.block-container {
    max-width: 1250px;
    padding-top: 2rem;
}
[data-testid="stMetric"] {
    background: white;
    border: 1px solid #dbe3ef;
    padding: 18px;
    border-radius: 14px;
}
[data-testid="stMetricValue"] {
    color: #2457a7;
}
[data-testid="stSidebar"] {
    background-color: #e8eef8;
}
</style>
""", unsafe_allow_html=True)

# ---------------- DASHBOARD ----------------
if page == "Dashboard":

    st.title("🚗 Welcome to AutoPrice AI")
    st.write(
        "Explore used car prices, market trends, "
        "and estimated selling prices."
    )

    prices = data["selling_price"].dropna()

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Total Records", f"{len(data):,}")
    c2.metric(
        "Average Price",
        f"₹{prices.mean():,.0f}" if not prices.empty else "N/A"
    )
    c3.metric(
        "Highest Price",
        f"₹{prices.max():,.0f}" if not prices.empty else "N/A"
    )
    c4.metric("Car Models", f"{data['name'].nunique():,}")

    st.divider()

    left, right = st.columns(2)

    with left:
        st.subheader("Average Price by Fuel Type")
        fuel_prices = data.dropna(
            subset=["fuel", "selling_price"]
        ).groupby("fuel")["selling_price"].mean()
        st.bar_chart(fuel_prices)

    with right:
        st.subheader("Average Price by Year")
        year_prices = data.dropna(
            subset=["year", "selling_price"]
        ).groupby("year")["selling_price"].mean()
        st.line_chart(year_prices)

    st.subheader("Top 10 Car Models")
    st.bar_chart(data["name"].value_counts().head(10))

    st.info(
        "Predicted prices are estimates and may differ "
        "from actual market prices."
    )

# ---------------- PRICE PREDICTION ----------------
elif page == "Price Prediction":

    st.title("🚘 Used Car Price Prediction")

    names = sorted(data["name"].dropna().unique())
    selected_name = st.selectbox("Select Car Model", names)

    rows = data[data["name"] == selected_name].sort_values(
        "year", ascending=False
    )
    car_row = rows.iloc[0]

    c1, c2, c3 = st.columns(3)

    with c1:
        year_default = int(car_row["year"]) if pd.notna(car_row["year"]) else 2018
        year = st.number_input(
            "Manufacturing Year",
            min_value=1990,
            max_value=2026,
            value=max(1990, min(2026, year_default))
        )

        km_default = int(car_row["km_driven"]) if pd.notna(car_row["km_driven"]) else 30000
        km_driven = st.number_input(
            "Kilometers Driven",
            min_value=0,
            max_value=1000000,
            value=max(0, min(1000000, km_default)),
            step=1000
        )

        mileage_default = float(car_row["mileage"]) if pd.notna(car_row["mileage"]) else 20.0
        mileage = st.number_input(
            "Mileage (km/l)",
            min_value=0.0,
            max_value=100.0,
            value=max(0.0, min(100.0, mileage_default))
        )

    with c2:
        fuel_options = sorted(data["fuel"].dropna().unique().tolist())
        fuel = st.selectbox(
            "Fuel Type",
            fuel_options,
            index=fuel_options.index(car_row["fuel"])
            if car_row["fuel"] in fuel_options else 0
        )

        seller_options = sorted(data["seller_type"].dropna().unique().tolist())
        seller_type = st.selectbox(
            "Seller Type",
            seller_options,
            index=seller_options.index(car_row["seller_type"])
            if car_row["seller_type"] in seller_options else 0
        )

        transmission_options = sorted(data["transmission"].dropna().unique().tolist())
        transmission = st.selectbox(
            "Transmission",
            transmission_options,
            index=transmission_options.index(car_row["transmission"])
            if car_row["transmission"] in transmission_options else 0
        )

    with c3:
        owner_options = sorted(data["owner"].dropna().unique().tolist())
        owner = st.selectbox(
            "Previous Owners",
            owner_options,
            index=owner_options.index(car_row["owner"])
            if car_row["owner"] in owner_options else 0
        )

        engine_default = float(car_row["engine"]) if pd.notna(car_row["engine"]) else 1200.0
        engine = st.number_input(
            "Engine (CC)",
            min_value=0.0,
            max_value=10000.0,
            value=max(0.0, min(10000.0, engine_default))
        )

        power_default = float(car_row["max_power"]) if pd.notna(car_row["max_power"]) else 80.0
        max_power = st.number_input(
            "Max Power (bhp)",
            min_value=0.0,
            max_value=2000.0,
            value=max(0.0, min(2000.0, power_default))
        )

    seats_options = sorted(
        int(s) for s in rows["seats"].dropna().unique()
        if 1 <= s <= 20
    )
    if not seats_options:
        seats_options = [4, 5, 6, 7]

    seats_default = (
        int(car_row["seats"])
        if pd.notna(car_row["seats"]) and int(car_row["seats"]) in seats_options
        else seats_options[0]
    )
    seats = st.selectbox(
        "Seats", seats_options,
        index=seats_options.index(seats_default)
    )

    if st.button("✨ Predict Car Price", use_container_width=True):

        car = pd.DataFrame([{
            "name": selected_name,
            "year": year,
            "km_driven": km_driven,
            "fuel": fuel,
            "seller_type": seller_type,
            "transmission": transmission,
            "owner": owner,
            "mileage": mileage,
            "engine": engine,
            "max_power": max_power,
            "seats": seats
        }])

        try:
            prediction = float(model.predict(car)[0])

            st.metric("Estimated Selling Price", f"₹{prediction:,.0f}")

            new_entry = pd.DataFrame([{
                "Car Model": selected_name,
                "Predicted Price": round(prediction, 2),
                "Date": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
            }])

            history = pd.concat([history, new_entry], ignore_index=True)
            history.to_csv(history_file, index=False)
            st.success("Prediction saved to history.")

        except Exception as e:
            st.error(f"Prediction error: {e}")

# ---------------- ANALYTICS ----------------
elif page == "Analytics":

    st.title("📈 Car Market Analytics")

    st.subheader("Price Distribution")
    st.histogram if False else None

    prices = data["selling_price"].dropna()
    if not prices.empty:
        histogram_data = pd.cut(prices, bins=20).value_counts().sort_index()
        histogram_data.index = histogram_data.index.astype(str)
        st.bar_chart(histogram_data)

    st.subheader("Average Price by Fuel Type")
    st.bar_chart(
        data.dropna(subset=["fuel", "selling_price"])
        .groupby("fuel")["selling_price"].mean()
    )

    st.subheader("Average Price by Manufacturing Year")
    st.line_chart(
        data.dropna(subset=["year", "selling_price"])
        .groupby("year")["selling_price"].mean()
    )

# ---------------- PREDICTION HISTORY ----------------
elif page == "Prediction History":

    st.title("📜 Prediction History")

    if os.path.exists(history_file):
        history = pd.read_csv(history_file)

    if history.empty:
        st.info("No prediction history yet.")
    else:
        st.dataframe(history, use_container_width=True)

        chart_data = history.copy()
        chart_data["Predicted Price"] = pd.to_numeric(
            chart_data["Predicted Price"], errors="coerce"
        )
        chart_data = chart_data.dropna(subset=["Predicted Price"])

        if not chart_data.empty:
            st.bar_chart(chart_data.set_index("Car Model")["Predicted Price"])

        st.download_button(
            "⬇️ Download History CSV",
            data=history.to_csv(index=False).encode("utf-8"),
            file_name="prediction_history.csv",
            mime="text/csv"
        )

        if st.button("🗑️ Clear Prediction History"):
            history = pd.DataFrame(
                columns=["Car Model", "Predicted Price", "Date"]
            )
            history.to_csv(history_file, index=False)
            st.success("History cleared.")
            st.rerun()

# ---------------- CAR PRICE COMPARISON ----------------
elif page == "Car Price Comparison":

    st.title("🚗 Car Price Comparison")

    car_names = sorted(data["name"].dropna().unique())

    if len(car_names) < 2:
        st.warning("At least two car models are needed.")
    else:
        c1, c2 = st.columns(2)

        with c1:
            car1 = st.selectbox("First Car", car_names, key="car1")

        with c2:
            car2 = st.selectbox(
                "Second Car", car_names,
                index=1, key="car2"
            )

        if st.button("Compare Cars"):
            try:
                def get_prediction(car_name):
                    rows = data[data["name"] == car_name].dropna(
                        subset=[
                            "year", "km_driven", "fuel", "seller_type",
                            "transmission", "owner", "mileage",
                            "engine", "max_power", "seats"
                        ]
                    ).sort_values("year", ascending=False)

                    if rows.empty:
                        raise ValueError(f"Incomplete details for {car_name}")

                    row = rows.iloc[0]

                    inputs = pd.DataFrame([{
                        "name": car_name,
                        "year": row["year"],
                        "km_driven": row["km_driven"],
                        "fuel": row["fuel"],
                        "seller_type": row["seller_type"],
                        "transmission": row["transmission"],
                        "owner": row["owner"],
                        "mileage": row["mileage"],
                        "engine": row["engine"],
                        "max_power": row["max_power"],
                        "seats": row["seats"]
                    }])

                    return float(model.predict(inputs)[0]), row

                price1, row1 = get_prediction(car1)
                price2, row2 = get_prediction(car2)

                c1, c2 = st.columns(2)
                c1.metric(car1, f"₹{price1:,.0f}")
                c2.metric(car2, f"₹{price2:,.0f}")

                st.bar_chart(pd.Series({car1: price1, car2: price2}))
                st.info(f"Estimated price difference: ₹{abs(price1-price2):,.0f}")

                if price1 < price2:
                    st.success(f"{car1} has the lower estimated price.")
                elif price2 < price1:
                    st.success(f"{car2} has the lower estimated price.")
                else:
                    st.info("Both estimates are equal.")

            except Exception as e:
                st.error(f"Comparison error: {e}")

# ---------------- MODEL PERFORMANCE ----------------
elif page == "Model Performance":

    st.title("🎯 Model Performance")
    st.write("Evaluation of a model trained on a held-out test split.")

    feature_columns = [
        "name", "year", "km_driven", "fuel", "seller_type",
        "transmission", "owner", "mileage", "engine",
        "max_power", "seats"
    ]
    target_column = "selling_price"

    evaluation_data = data[feature_columns + [target_column]].copy()
    evaluation_data = evaluation_data.dropna(subset=[target_column])

    if len(evaluation_data) < 10:
        st.warning("Not enough valid rows to evaluate the model.")
    else:
        X = evaluation_data[feature_columns]
        y = evaluation_data[target_column]

        numeric_features = [
            "year", "km_driven", "mileage", "engine",
            "max_power", "seats"
        ]
        categorical_features = [
            "name", "fuel", "seller_type", "transmission", "owner"
        ]

        numeric_transformer = Pipeline([
            ("imputer", SimpleImputer(strategy="median"))
        ])

        categorical_transformer = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore"))
        ])

        preprocessor = ColumnTransformer([
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features)
        ])

        evaluation_model = Pipeline([
            ("preprocessor", preprocessor),
            ("regressor", RandomForestRegressor(
                n_estimators=100,
                random_state=42,
                n_jobs=-1
            ))
        ])

        if st.button("Calculate Model Performance"):
            try:
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42
                )

                with st.spinner("Training and evaluating model..."):
                    evaluation_model.fit(X_train, y_train)
                    predictions = evaluation_model.predict(X_test)

                r2 = r2_score(y_test, predictions)
                mae = mean_absolute_error(y_test, predictions)
                rmse = np.sqrt(mean_squared_error(y_test, predictions))

                c1, c2, c3 = st.columns(3)
                c1.metric("R² Score", f"{r2:.3f}")
                c2.metric("MAE", f"₹{mae:,.0f}")
                c3.metric("RMSE", f"₹{rmse:,.0f}")

                st.write(
                    "**R² Score:** How much of the price variation "
                    "the evaluation model explains. Closer to 1 is generally better."
                )
                st.write(
                    "**MAE:** Average absolute difference between "
                    "predicted and actual prices. Lower is better."
                )
                st.write(
                    "**RMSE:** Penalizes larger prediction errors more. Lower is better."
                )

                result_data = pd.DataFrame({
                    "Actual Price": y_test.to_numpy(),
                    "Predicted Price": predictions
                })
                st.subheader("Actual vs Predicted Prices")
                st.scatter_chart(
                    result_data,
                    x="Actual Price",
                    y="Predicted Price"
                )

                st.warning(
                    "These metrics evaluate a newly trained Random Forest model "
                    "on a held-out split. They do not measure the saved "
                    "car_price_model.pkl unless it uses the same training procedure."
                )

            except Exception as e:
                st.error(f"Evaluation error: {e}")

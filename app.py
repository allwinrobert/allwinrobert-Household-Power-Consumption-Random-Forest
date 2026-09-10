
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Household Power Consumption",
    page_icon="⚡",
    layout="wide"
)

# --------------------------------------------------
# PATHS
# --------------------------------------------------

MODEL_PATH = os.path.join(
    os.path.dirname(__file__),
    "model",
    "random_forest_model.pkl"
)

DATA_PATH = os.path.join(
    os.path.dirname(__file__),
    "data",
    "household_power_consumption_cleaned_minute.csv.gz"
)

# --------------------------------------------------
# LOAD MODEL
# --------------------------------------------------

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

model = load_model()

# --------------------------------------------------
# FEATURES
# --------------------------------------------------

features = [
    "Global_reactive_power",
    "Voltage",
    "Global_intensity",
    "Sub_metering_1",
    "Sub_metering_2",
    "Sub_metering_3",
    "hour",
    "day",
    "month",
    "day_of_week",
    "is_weekend"
]

# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

st.sidebar.title("⚡ Household Power")
st.sidebar.markdown("### Random Forest Prediction")

page = st.sidebar.radio(
    "Navigation",
    [
        "🏠 Home",
        "📊 Dataset Overview",
        "🔮 Power Prediction",
        "📈 Model Performance",
        "🌲 Feature Importance"
    ]
)

# --------------------------------------------------
# HOME
# --------------------------------------------------

if page == "🏠 Home":

    st.title("⚡ Household Power Consumption Prediction")

    st.markdown(
        """
        ### Random Forest Machine Learning Project

        This application predicts **Global Active Power**
        using household electrical measurements and
        time-based features.

        The prediction model is built using a
        **Random Forest Regressor**.
        """
    )

    st.divider()

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Dataset Records",
            "2,075,259"
        )

    with col2:
        st.metric(
            "Input Features",
            "11"
        )

    with col3:
        st.metric(
            "R² Score",
            "0.9985"
        )

    st.divider()

    st.subheader("Project Workflow")

    st.markdown(
        """
        **Dataset → Data Cleaning → Feature Engineering
        → Train/Test Split → Random Forest → Prediction**
        """
    )

    st.info(
        "The model was trained using 300,000 representative "
        "training records and evaluated on the later test period."
    )


# --------------------------------------------------
# DATASET OVERVIEW
# --------------------------------------------------

elif page == "📊 Dataset Overview":

    st.title("📊 Dataset Overview")

    st.write(
        "Household power consumption dataset containing "
        "minute-level electrical measurements."
    )

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Total Records", "2,075,259")

    with col2:
        st.metric("Number of Columns", "9")

    st.subheader("Dataset Columns")

    columns_df = pd.DataFrame({
        "Column": [
            "datetime",
            "Global_active_power",
            "Global_reactive_power",
            "Voltage",
            "Global_intensity",
            "Sub_metering_1",
            "Sub_metering_2",
            "Sub_metering_3",
            "was_imputed"
        ]
    })

    st.dataframe(
        columns_df,
        use_container_width=True,
        hide_index=True
    )

    st.subheader("Sample Data")

    try:
        sample = pd.read_csv(
            DATA_PATH,
            nrows=1000
        )

        st.dataframe(
            sample.head(20),
            use_container_width=True
        )

    except Exception as e:
        st.error(f"Unable to load dataset: {e}")


# --------------------------------------------------
# POWER PREDICTION
# --------------------------------------------------

elif page == "🔮 Power Prediction":

    st.title("🔮 Power Consumption Prediction")

    st.write(
        "Enter the household electrical measurements "
        "and time information below."
    )

    st.divider()

    col1, col2 = st.columns(2)

    with col1:

        reactive_power = st.number_input(
            "Global Reactive Power",
            min_value=0.0,
            max_value=10.0,
            value=0.2,
            step=0.01
        )

        voltage = st.number_input(
            "Voltage",
            min_value=0.0,
            max_value=300.0,
            value=240.0,
            step=0.1
        )

        intensity = st.number_input(
            "Global Intensity",
            min_value=0.0,
            max_value=100.0,
            value=10.0,
            step=0.1
        )

        sub_metering_1 = st.number_input(
            "Sub Metering 1",
            min_value=0.0,
            value=0.0,
            step=0.1
        )

        sub_metering_2 = st.number_input(
            "Sub Metering 2",
            min_value=0.0,
            value=1.0,
            step=0.1
        )

        sub_metering_3 = st.number_input(
            "Sub Metering 3",
            min_value=0.0,
            value=10.0,
            step=0.1
        )

    with col2:

        hour = st.slider(
            "Hour",
            min_value=0,
            max_value=23,
            value=12
        )

        day = st.slider(
            "Day",
            min_value=1,
            max_value=31,
            value=15
        )

        month = st.slider(
            "Month",
            min_value=1,
            max_value=12,
            value=6
        )

        day_of_week = st.slider(
            "Day of Week",
            min_value=0,
            max_value=6,
            value=2,
            help="0 = Monday, 6 = Sunday"
        )

        is_weekend = st.selectbox(
            "Weekend?",
            ["No", "Yes"]
        )

        weekend_value = 1 if is_weekend == "Yes" else 0

    st.divider()

    if st.button(
        "⚡ Predict Power Consumption",
        use_container_width=True
    ):

        input_data = pd.DataFrame([[
            reactive_power,
            voltage,
            intensity,
            sub_metering_1,
            sub_metering_2,
            sub_metering_3,
            hour,
            day,
            month,
            day_of_week,
            weekend_value
        ]], columns=features)

        prediction = model.predict(input_data)[0]

        st.success("Prediction completed!")

        st.metric(
            "Predicted Global Active Power",
            f"{prediction:.3f} kW"
        )

        st.caption(
            "Prediction generated by the trained Random Forest model."
        )


# --------------------------------------------------
# MODEL PERFORMANCE
# --------------------------------------------------

elif page == "📈 Model Performance":

    st.title("📈 Random Forest Model Performance")

    st.write(
        "Performance of the Random Forest Regressor "
        "on the held-out test period."
    )

    st.divider()

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("MAE", "0.0209")

    with col2:
        st.metric("MSE", "0.0012")

    with col3:
        st.metric("RMSE", "0.0342")

    with col4:
        st.metric("R² Score", "0.9985")

    st.divider()

    st.subheader("Metric Interpretation")

    st.markdown(
        """
        **MAE:** Average absolute prediction error.

        **MSE:** Average squared prediction error.

        **RMSE:** Square root of the mean squared error.

        **R² Score:** Proportion of target variance explained
        by the model.
        """
    )

    st.info(
        "The very high R² score is influenced by the strong "
        "relationship between Global Intensity and Global Active Power."
    )


# --------------------------------------------------
# FEATURE IMPORTANCE
# --------------------------------------------------

elif page == "🌲 Feature Importance":

    st.title("🌲 Random Forest Feature Importance")

    importance_df = pd.DataFrame({
        "Feature": features,
        "Importance": model.feature_importances_
    })

    importance_df = importance_df.sort_values(
        "Importance",
        ascending=False
    )

    st.subheader("Feature Importance")

    st.bar_chart(
        importance_df.set_index("Feature")
    )

    st.subheader("Importance Table")

    st.dataframe(
        importance_df,
        use_container_width=True,
        hide_index=True
    )

    st.info(
        "Global Intensity is the dominant feature in this model."
    )

# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.sidebar.divider()
st.sidebar.caption("Random Forest | Streamlit ML Project")

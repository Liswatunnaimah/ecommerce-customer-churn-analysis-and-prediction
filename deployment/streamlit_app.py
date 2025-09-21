import streamlit as st
import pandas as pd
import pickle

# --- APP CONFIGURATION ---
st.set_page_config(page_title="E-Commerce Churn Predictor", layout="centered")

# --- HEADER ---
st.title("E-Commerce Churn Prediction App")
st.markdown("""
This application estimates the likelihood that a customer will **churn**, meaning they may stop using the platform or making purchases.  


**Instructions**: Enter customer details in the sidebar and click **Predict** to assess churn risk.
""")

# --- LOAD MODEL ---
@st.cache_resource
def load_model(pickle_path='final_model.pkl'):
    try:
        with open(pickle_path, 'rb') as f:
            return pickle.load(f)
    except FileNotFoundError:
        st.error("Model file not found. Please contact the administrator.")
        return None

model = load_model()

# --- USER INPUT ---
if model is not None:
    st.sidebar.header("CUSTOMER DETAILS")
    st.markdown("<br>", unsafe_allow_html=True)

    with st.sidebar:
        # --- Section 1: Customer Engagement ---
        st.subheader("Customer Engagement")

        tenure = st.number_input('Tenure (months)', 0, 60, 12)

        # --- Tenure Group (Feature Engineering) ---
        if tenure <= 3:
            tenure_group = 'New'
        elif tenure <= 9:
            tenure_group = 'Early'
        elif tenure <= 15:
            tenure_group = 'MidTerm'
        else:
            tenure_group = 'LongTerm'
        st.markdown(f"Tenure Group: `{tenure_group}`")

        hour_spend_on_app = st.slider('Hours on App/Week', 0.0, 5.0, 1.0)
        num_device_registered = st.number_input('Devices Registered', 1, 10, 2)
        days_since_last_order = st.number_input('Days Since Last Order', 0, 100, 10)

        # --- Recency Ratio & Active User (Feature Engineering) ---
        recency_ratio = days_since_last_order / (tenure + 1)
        st.markdown(f"Recency Ratio: {recency_ratio:.2f}")

        is_active_user = int(days_since_last_order < 30)
        st.markdown(f"Active User: {'Yes' if is_active_user else 'No'}")

        st.markdown("<br>", unsafe_allow_html=True)

        # --- Section 2: Customer Profile ---
        st.subheader("Customer Profile")
        city_tier = st.selectbox('City Tier', [1, 2, 3])
        gender = st.selectbox('Gender', ['Female', 'Male'])
        marital_status = st.selectbox('Marital Status', ['Single', 'Divorced', 'Married'])

        st.markdown("<br>", unsafe_allow_html=True)

        # --- Section 3: Behavioral Preferences ---
        st.subheader("Behavioral Preferences")
        preferred_login_device = st.selectbox('Login Device', ['Mobile Phone', 'Computer'])
        preferred_payment_mode = st.selectbox('Payment Mode', ['Debit Card', 'UPI', 'Credit Card', 'Cash on Delivery', 'E wallet'])
        preferred_order_cat = st.selectbox('Preferred Category', ['Laptop & Accessory', 'Mobile Phone', 'Others', 'Fashion', 'Grocery'])
        warehouse_to_home = st.number_input('Distance to Warehouse (km)', 0.0, 100.0, 10.0)
        number_of_address = st.number_input('Number of Addresses', 1, 10, 1)

        st.markdown("<br>", unsafe_allow_html=True)

        # --- Section 4: Transactions and Feedback ---
        st.subheader("Transactions and Feedback")
        order_count = st.number_input('Order Count', 0, 50, 1)
        order_amount_hike = st.number_input('Order Amount Hike (%)', 0.0, 100.0, 10.0)
        coupon_used = st.number_input('Coupons Used', 0, 20, 0)
        cashback_amount = st.number_input('Cashback Amount', 0, 500, 0)
        satisfaction_score = st.slider('Satisfaction Score', 1, 5, 3)
        complain = st.selectbox('Has Complaints?', ['No', 'Yes']) == 'Yes'

        # --- Feature Engineering: Unhappy Customer ---
        unhappy_customer = int(complain and satisfaction_score <= 2)
        st.markdown(f"Unhappy Customer: {'Yes' if unhappy_customer else 'No'}")

        st.markdown("<br>", unsafe_allow_html=True)

    # --- FORMAT INPUT ---
    input_dict = {
        'Tenure': tenure,
        'PreferredLoginDevice': preferred_login_device,
        'CityTier': city_tier,
        'WarehouseToHome': warehouse_to_home,
        'PreferredPaymentMode': preferred_payment_mode,
        'Gender': gender,
        'HourSpendOnApp': hour_spend_on_app,
        'NumberOfDeviceRegistered': num_device_registered,
        'PreferedOrderCat': preferred_order_cat,
        'SatisfactionScore': satisfaction_score,
        'MaritalStatus': marital_status,
        'NumberOfAddress': number_of_address,
        'Complain': int(complain),
        'OrderAmountHikeFromlastYear': order_amount_hike,
        'CouponUsed': coupon_used,
        'OrderCount': order_count,
        'DaySinceLastOrder': days_since_last_order,
        'CashbackAmount': cashback_amount,
        'RecencyRatio': recency_ratio,
        'IsActiveUser': is_active_user,
        'UnhappyCustomer': unhappy_customer,
        'TenureGroup': tenure_group
    }

    input_df = pd.DataFrame([input_dict])

    # --- PREDICTION ---
    if st.button('Predict'):
        prediction = model.predict(input_df)[0]
        proba = model.predict_proba(input_df)[0][1]

        proba_pct = proba * 100
        stay_pct = (1 - proba) * 100

        if prediction == 1:
            st.error(f"Churn Prediction: **Yes** (Probability: {proba_pct:.2f}%)")
            st.markdown("**Action Needed:** This customer is at risk of leaving. Consider targeted engagement.")
        else:
            st.success(f"Churn Prediction: **No** (Probability of Staying: {stay_pct:.2f}%)")
            st.markdown("**Good News:** This customer is likely to remain loyal.")

# --- IMAGE ---
st.image("ecommercepict.png", use_container_width=True)

# --- ABOUT ---
with st.expander("ℹ️ About This App"):
    st.markdown("""
This prediction tool helps e-commerce teams **identify high-risk customers** based on their behavior, satisfaction, and engagement level.

- Built using a supervised ML model trained on real e-commerce customer data.
- Use this tool to **prioritize retention campaigns**, improve support, and target offers.

**Churn** here means the customer is unlikely to continue using the app or shopping on the platform.
    """)

# --- FEATURE EXPLANATIONS ---
with st.expander("ℹ️ Feature Explanations"):
    st.markdown("""
- **`Tenure`**: Duration as a customer (**months**).
- **`TenureGroup`**: Loyalty bin based on tenure: `New (≤3)`, `Early (4–9)`, `MidTerm (10–15)`, `LongTerm (>15)`.
- **`HourSpendOnApp`**: Avg. weekly app usage time (in hours).          
- **`NumberOfDeviceRegistered`**, **`NumberOfAddress`**: Device or address count in profile.
- **`DaySinceLastOrder`**: Days since most recent order.
- **`RecencyRatio`**: `DaySinceLastOrder ÷ (Tenure + 1)` — lower = more active.
- **`IsActiveUser`**: `Yes` if last order < 30 days ago, else `No`.
- **`CityTier`**: Customer's city level – `1`, `2`, or `3`.
- **`Gender`**, **`MaritalStatus`**: Self-explanatory demographic info.
- **`PreferredLoginDevice`**: Common login device – `Mobile Phone` or `Computer`.
- **`PreferredPaymentMode`**: Most used payment method.
- **`PreferedOrderCat`**: Top product category ordered last month.
- **`WarehouseToHome`**: Estimated distance from warehouse to home (**km**).
- **`OrderAmountHikeFromlastYear`**: % order increase from last year.          
- **`CouponUsed`**, **`OrderCount`**, **`CashbackAmount`**: Purchase-related metrics.
- **`SatisfactionScore`**: Rating from 1 (low) to 5 (high).
- **`Complain`**: `Yes` if complaint filed last month, `No` if not.
- **`UnhappyCustomer`**: Flag = `Yes` if complaint + satisfaction ≤ 2.
""")

# --- CONTRIBUTORS ---
with st.expander("ℹ️ Contributors"):
    st.markdown("""
- Zulfi Nadhia Cahyani (https://github.com/znadhiac)  
- Liswatun Naimah (https://github.com/Liswatunnaimah)  
- Aldino Dian Mandala Putra (https://github.com/aldino9112)  
    """)
import streamlit as st
import pandas as pd
import numpy as np
from prediction_function import predict_from_dataframe

# --- Streamlit Page Config ---
st.set_page_config(
    page_title="Fraud Detection Dashboard",
    page_icon="💳",
    layout="wide"
)

# --- Custom CSS Styling ---
st.markdown("""
<style>
body {
    background-color: #f8f9fa;
}
.main {
    padding: 1.5rem 3rem;
}
h1, h2, h3 {
    color: #1f2937;
}
.stButton>button {
    background: linear-gradient(to right, #2563eb, #1d4ed8);
    color: white;
    border-radius: 10px;
    border: none;
    padding: 0.75rem 1.5rem;
    font-size: 1rem;
    transition: 0.3s ease;
}
.stButton>button:hover {
    background: linear-gradient(to right, #1e40af, #1d4ed8);
    transform: scale(1.02);
}
.dataframe th {
    background-color: #2563eb !important;
    color: white !important;
}
.metric-card {
    background-color: white;
    border-radius: 15px;
    padding: 1.2rem;
    box-shadow: 0 2px 8px rgba(0,0,0,0.05);
    text-align: center;
}
.metric-value {
    font-size: 2rem;
    font-weight: 600;
    color: #2563eb;
}
.metric-label {
    color: #6b7280;
}
</style>
""", unsafe_allow_html=True)

# --- Header Section ---
st.title("💳 Intelligent Fraud Detection System")
st.write("Upload transaction data to analyze and detect **risky or fraudulent transactions** in real-time using AI.")

# --- File Upload Section ---
uploaded_file = st.file_uploader("📂 Upload Transaction CSV File", type=["csv"])

# --- Default model path ---
model_path = "gcn_correlation_smote_model.pth"

# --- Sample CSV download ---
sample_data = pd.DataFrame({
    "amount": [120.0, 5600.0, 23.5, 400.0],
    "time_seconds": [1000, 200000, 25000, 3600],
    "feature_a": [0.1, -1.2, 0.3, 0.0],
    "feature_b": [1, 0, 1, 1]
})
st.download_button("📄 Download Sample CSV", sample_data.to_csv(index=False).encode("utf-8"),
                   "sample_transactions.csv", "text/csv")

# --- Process the Uploaded CSV ---
if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        st.subheader("📊 Uploaded Data Preview")
        st.dataframe(df.head(8))
    except Exception as e:
        st.error(f"❌ Failed to load CSV: {e}")
        st.stop()

    # --- Run Prediction ---
    if st.button("🚀 Run Fraud Detection", use_container_width=True):
        with st.spinner("Analyzing transactions..."):
            try:
                out_df, meta = predict_from_dataframe(
                    df,
                    model_path=model_path,
                    use_gnn=True,
                    knn_k=5,
                    threshold=0.5,
                    device='cpu'
                )
            except Exception as e:
                st.error(f"⚠️ Prediction failed: {e}")
                st.stop()

        st.success(f"✅ Detection completed using method: {meta.get('used_method', 'Unknown')}")

        # --- Summary Metrics ---
        st.subheader("📈 Detection Summary")

        fraud_count = (out_df['Prediction_Label'] == 'Fraud').sum()
        total_count = len(out_df)
        fraud_rate = (fraud_count / total_count) * 100

        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown(f"<div class='metric-card'><div class='metric-value'>{total_count}</div><div class='metric-label'>Total Transactions</div></div>", unsafe_allow_html=True)
        with c2:
            st.markdown(f"<div class='metric-card'><div class='metric-value'>{fraud_count}</div><div class='metric-label'>Fraudulent Transactions</div></div>", unsafe_allow_html=True)
        with c3:
            st.markdown(f"<div class='metric-card'><div class='metric-value'>{fraud_rate:.2f}%</div><div class='metric-label'>Fraud Rate</div></div>", unsafe_allow_html=True)

        # --- Probability Distribution ---
        st.subheader("🔎 Fraud Probability Distribution")
        st.bar_chart(out_df['Fraud_Probability'].value_counts(bins=20).sort_index())

        # --- Risky Transactions ---
        risky = out_df[out_df['Prediction_Label'] == 'Fraud'].sort_values('Fraud_Probability', ascending=False)
        st.subheader("🚨 Risky Transactions Detected")
        if not risky.empty:
            st.dataframe(risky.head(50))
        else:
            st.success("🎉 No risky transactions detected at this threshold.")

        # --- Download Results ---
        csv = out_df.to_csv(index=False).encode("utf-8")
        st.download_button("💾 Download Full Prediction Report", csv, "fraud_predictions.csv", "text/csv")

else:
    st.info("⬆️ Upload a CSV file to begin fraud analysis.")

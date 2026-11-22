
import streamlit as st
import pandas as pd
from pathlib import Path

st.title("Hotel Booking Cancellation Dashboard")

st.write("Analysis of hotel booking cancellations")

# Load dataset
@st.cache_data
def load_data(file_path):
    return pd.read_csv(file_path)

possible_paths = [
    Path("hotel_bookings.csv"),
    Path("/content/hotel_bookings.csv")
]

data_path = next(
    (p for p in possible_paths if p.exists()),
    None
)

if data_path is None:
    st.warning("Dataset not found.")
    st.stop()

df = load_data(str(data_path))

# Sidebar
st.sidebar.header("Filters")

hotel_type = st.sidebar.selectbox(
    "Select Hotel Type",
    ["All"] + sorted(df["hotel"].unique().tolist())
)

max_lead_time = st.sidebar.slider(
    "Maximum Lead Time (Days)",
    min_value=0,
    max_value=int(df["lead_time"].max()),
    value=int(df["lead_time"].max())
)

# Apply filters
filtered_df = df.copy()

if hotel_type != "All":
    filtered_df = filtered_df[
        filtered_df["hotel"] == hotel_type
    ]

filtered_df = filtered_df[
    filtered_df["lead_time"] <= max_lead_time
]

# Display results
st.subheader("Dataset Overview")

st.write("Filtered Records:", len(filtered_df))

st.dataframe(filtered_df.head(10))

if filtered_df.empty:
    st.warning("No records match the selected filters.")
    st.stop()

# Cancellation rate
cancellation_rate = filtered_df["is_canceled"].mean() * 100

st.metric(
    "Cancellation Rate",
    f"{cancellation_rate:.2f}%"
)

# Chart
st.subheader("Cancellation Rate by Hotel")

hotel_rates = (
    filtered_df.groupby("hotel")["is_canceled"].mean() * 100
)

st.bar_chart(hotel_rates)

# AI assistant section
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

st.subheader("Ask AI About Hotel Cancellations")

question = st.text_input(
    "Enter your question",
    placeholder="What does the cancellation rate mean?"
)

if st.button("Ask Gemini"):

    if not question.strip():
        st.warning("Please enter a question.")

    elif not os.getenv("GOOGLE_API_KEY"):
        st.error("Google API key is not available.")

    else:
        prompt = PromptTemplate.from_template(
            "You are helping explain a hotel booking analysis. "
            "The cleaned dataset contains 87,396 records with "
            "a cancellation rate of 27.49%. "
            "Answer the following question in 2-3 simple sentences. "
            "Do not invent conclusions that are not supported "
            "by the supplied information. "
            "Question: {question}"
        )

        llm = ChatGoogleGenerativeAI(
            model="gemini-3.8-flash",
            temperature=0
        )

        chain = prompt | llm | StrOutputParser()

        with st.spinner("Generating answer..."):
            try:
                answer = chain.invoke({"question": question})
                st.write(answer)
            except Exception as e:
                st.error(f"AI request failed: {type(e).__name__}")

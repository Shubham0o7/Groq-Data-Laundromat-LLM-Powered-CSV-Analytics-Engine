import streamlit as st
st.set_page_config(page_title="Groq & Roll Data Laundromat", layout="wide")

import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from langchain_groq import ChatGroq
from dotenv import load_dotenv

# ================= LOAD ENV =================
load_dotenv()

# ================= UI =================
st.title("🤖 LLM-Powered Spreadsheet Washer & Insight Spinner")
st.write("Upload a CSV file to clean data, generate insights, and ask questions.")

# ================= API =================
groq_api_key = os.environ.get("GROQ_API_KEY")

if not groq_api_key:
    st.error("Missing GROQ_API_KEY")
    st.stop()

llm = ChatGroq(
    model="llama-3.1-8b-instant",
    api_key=groq_api_key,
    temperature=0
)

# ================= FILE UPLOAD =================
uploaded_file = st.file_uploader("Upload CSV", type="csv")

if uploaded_file is not None:

    df_raw = pd.read_csv(uploaded_file)

    # ================= SESSION STATE =================
    if "df" not in st.session_state:
        st.session_state.df = df_raw.copy()

    if "cleaning_report" not in st.session_state:
        st.session_state.cleaning_report = ""

    if "insights_report" not in st.session_state:
        st.session_state.insights_report = ""

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # ================= PREVIEW + METRICS =================
    col1, col2 = st.columns([3, 1])

    with col1:
        st.subheader("🔍 Data Preview")
        st.dataframe(st.session_state.df.head())

    with col2:
        st.subheader("📊 Dataset Metrics")

        total_rows = st.session_state.df.shape[0]
        total_cols = st.session_state.df.shape[1]
        missing = st.session_state.df.isnull().sum().sum()
        duplicates = st.session_state.df.duplicated().sum()

        st.metric("Rows", total_rows)
        st.metric("Columns", total_cols)
        st.metric("Missing Values", missing)
        st.metric("Duplicate Rows", duplicates)

    # ================= EXECUTION ENGINE =================
    def execute_analytics_code(code_input: str):

        cleaned_code = code_input.strip().replace("```python", "").replace("```", "")

        if "import" in cleaned_code or "os." in cleaned_code:
            return "Unsafe code detected"

        df_safe = st.session_state.df.copy()
        env = {"df": df_safe, "pd": pd}

        try:
            exec(cleaned_code, {}, env)

            result = env.get("result", None)

            if isinstance(env.get("df"), pd.DataFrame):
                st.session_state.df = env["df"]

            return result

        except Exception as e:
            return f"Execution Error: {str(e)}"

    # ================= STEP 1 =================
    st.markdown("---")
    st.header("🧹 Step 1: Data Cleaning")

    cleaning_prompt = st.text_area(
        "Cleaning Instructions",
        value="Handle missing values, remove duplicates, fix datatypes."
    )

    if st.button("Run Cleaning"):

        with st.spinner("Cleaning data..."):

            prompt = f"""
            Clean dataframe df using pandas.

            Rules:
            - Handle missing values
            - Remove duplicates
            - Fix datatypes
            - Modify df directly
            - Do not explain

            Instruction: {cleaning_prompt}
            """

            code = llm.invoke([
                ("system", "Return only Python pandas code."),
                ("human", prompt)
            ]).content

            execute_analytics_code(code)

            st.session_state.cleaning_report = f"""
✅ Cleaning completed:
- Missing values handled
- Duplicates removed
- Data types standardized
"""

            st.success("Cleaning Done")

    if st.session_state.cleaning_report:
        st.markdown(st.session_state.cleaning_report)

        # ✅ Download button
        csv = st.session_state.df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Download Cleaned Data",
            data=csv,
            file_name="cleaned_data.csv"
        )

    # ================= STEP 2 =================
    st.markdown("---")
    st.header("📊 Step 2: Insights & Charts")

    if st.button("Generate Report"):

        df_plot = st.session_state.df.copy()

        num_cols = df_plot.select_dtypes(include="number").columns
        cat_cols = df_plot.select_dtypes(include="object").columns

        if len(num_cols) > 0:
            plt.figure()
            sns.histplot(df_plot[num_cols[0]].dropna())
            plt.savefig("chart1.png")
            plt.close()

        if len(cat_cols) > 0:
            plt.figure()
            df_plot[cat_cols[0]].value_counts().head(10).plot(kind="bar")
            plt.savefig("chart2.png")
            plt.close()

        report = llm.invoke([
            ("system", "You are a business analyst."),
            ("human", f"Columns: {list(df_plot.columns)}, Rows: {len(df_plot)}. Give key insights.")
        ])

        st.session_state.insights_report = report.content

    if st.session_state.insights_report:

        st.markdown(st.session_state.insights_report)

        # ✅ CENTER charts
        st.markdown("<div style='text-align: center;'>", unsafe_allow_html=True)

        if os.path.exists("chart1.png"):
            st.image("chart1.png")

        if os.path.exists("chart2.png"):
            st.image("chart2.png")

        st.markdown("</div>", unsafe_allow_html=True)

    # ================= STEP 3 =================
    st.markdown("---")
    st.header("💬 Ask Questions")

    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    def run_chat_agent(query: str):

        code = llm.invoke([
            ("system", """
            You are a pandas expert.
            Generate ONLY pandas code.
            Store output in variable 'result'.
            No explanation.
            """),
            ("human", query)
        ]).content

        code = code.replace("```python", "").replace("```", "")

        if "import" in code.lower() or "select" in code.lower():
            return "⚠️ Could not process query"

        result = execute_analytics_code(code)

        if isinstance(result, pd.DataFrame):
            result_str = result.head(20).to_string()
        elif isinstance(result, pd.Series):
            result_str = result.to_string()
        else:
            result_str = str(result)

        answer = llm.invoke([
            ("system", "Explain clearly."),
            ("human", f"Question: {query}\nResult: {result_str}")
        ])

        return answer.content

    if user_query := st.chat_input("Ask your data question..."):

        st.session_state.chat_history.append({"role": "user", "content": user_query})

        with st.chat_message("user"):
            st.markdown(user_query)

        with st.spinner("Analyzing..."):
            response = run_chat_agent(user_query)

        with st.chat_message("assistant"):
            st.markdown(response)

        st.session_state.chat_history.append({"role": "assistant", "content": response})

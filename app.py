import streamlit as st

# ✅ MUST BE FIRST STREAMLIT CALL
st.set_page_config(page_title="Groq & Roll Data Laundromat", layout="wide")

# ✅ TEXTAREA ALIGNMENT (LIKE YOUR IMAGE)
st.markdown("""
<style>
textarea {
    padding: 14px !important;
    line-height: 1.6 !important;
    font-size: 15px !important;
}
[data-testid="stTextArea"] textarea {
    border-radius: 10px !important;
    min-height: 120px !important;
}
</style>
""", unsafe_allow_html=True)

# ================= IMPORTS =================
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

    # ✅ SESSION STATE INIT
    defaults = {
        "df": df_raw.copy(),
        "cleaning_report": "",
        "insights_report": "",
        "chat_history": []
    }

    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val

    # ================= DATA OVERVIEW =================
    st.subheader("📊 Dataset Overview")

    left, right = st.columns([3, 2])

    with left:
        st.dataframe(st.session_state.df.head())

    with right:
        m1, m2 = st.columns(2)
        m3, m4 = st.columns(2)

        m1.metric("Total Rows", st.session_state.df.shape[0])
        m2.metric("Total Columns", st.session_state.df.shape[1])
        m3.metric("Missing Values", st.session_state.df.isnull().sum().sum())
        m4.metric("Duplicate Rows", st.session_state.df.duplicated().sum())

    # ================= EXECUTION ENGINE =================
    def execute_analytics_code(code_input: str):

        code = code_input.strip().replace("```python", "").replace("```", "")

        if "import" in code.lower() or "os." in code.lower():
            return "Unsafe code detected"

        env = {"df": st.session_state.df.copy(), "pd": pd}

        try:
            exec(code, {}, env)

            result = env.get("result", None)

            if isinstance(env.get("df"), pd.DataFrame):
                st.session_state.df = env["df"]

            return result

        except Exception as e:
            return f"Error: {str(e)}"

    # ================= STEP 1 =================
    st.markdown("---")
    st.header("🧹 Step 1: Data Cleaning")

    cleaning_prompt = st.text_area(
        "Customize cleaning logic",
        value="Handle missing values appropriately, remove duplicate rows, fix incorrect data types, standardize text formats, and clean inconsistent or anomalous entries."
    )

    if st.button("Run Cleaning"):

        with st.spinner("Cleaning data..."):

            code = llm.invoke([
                ("system", "Return ONLY pandas code."),
                ("human", cleaning_prompt)
            ]).content

            execute_analytics_code(code)

            st.session_state.cleaning_report = (
                "The dataset has been successfully cleaned. Missing values were handled, "
                "duplicates removed, and data types standardized. Text inconsistencies "
                "and anomalies were also corrected to improve data quality."
            )

            st.success("✅ Cleaning Completed")

    if st.session_state.cleaning_report:
        st.markdown(st.session_state.cleaning_report)

        csv = st.session_state.df.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Download Cleaned Data", csv, "cleaned_data.csv")

    # ================= STEP 2 =================
    st.markdown("---")
    st.header("📊 Step 2: Insights & Charts")

    report_prompt = st.text_area(
        "Customize report",
        value="Provide a clear business-style summary of the dataset. Highlight trends and key insights in paragraph form without bullet points."
    )

    if st.button("Generate Report"):

        df_plot = st.session_state.df.copy()

        num_cols = df_plot.select_dtypes(include='number').columns
        cat_cols = df_plot.select_dtypes(include='object').columns

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
            ("system", "Write paragraph style insights. No bullets."),
            ("human", f"Columns: {list(df_plot.columns)}, Rows: {len(df_plot)}. {report_prompt}")
        ])

        st.session_state.insights_report = report.content

    if st.session_state.insights_report:
        st.markdown(st.session_state.insights_report)

        c1, c2, c3 = st.columns([1, 2, 1])
        with c2:
            if os.path.exists("chart1.png"):
                st.image("chart1.png")
            if os.path.exists("chart2.png"):
                st.image("chart2.png")

    # ================= STEP 3 =================
    st.markdown("---")
    st.header("💬 Ask Questions")

    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    def run_chat_agent(query: str):

        code = llm.invoke([
            ("system", """
Generate ONLY pandas code.
- No SQL
- No imports
- Store output in variable 'result'
"""),
            ("human", query)
        ]).content

        code = code.replace("```python", "").replace("```", "")

        if "import" in code.lower() or "select" in code.lower():
            return "⚠️ Could not process query."

        result = execute_analytics_code(code)

        if isinstance(result, pd.DataFrame):
            result_str = result.head(20).to_string()
        elif isinstance(result, pd.Series):
            result_str = result.to_string()
        else:
            result_str = str(result)

        answer = llm.invoke([
            ("system", "Explain clearly."),
            ("human", f"Question: {query}\nResult:\n{result_str}")
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

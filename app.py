import streamlit as st  # ✅ MUST BE FIRST

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

    # ✅ Session state init
    if "df" not in st.session_state:
        st.session_state.df = df_raw.copy()

    if "cleaning_report" not in st.session_state:
        st.session_state.cleaning_report = ""

    if "insights_report" not in st.session_state:
        st.session_state.insights_report = ""

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # ================= PREVIEW =================
    st.subheader("Preview")
    st.dataframe(st.session_state.df.head())

    # ================= EXECUTION ENGINE =================
    def execute_analytics_code(code_input: str):
        cleaned_code = code_input.strip().replace("```python", "").replace("```", "")

        # ✅ safety
        if "import" in cleaned_code or "os." in cleaned_code:
            return "Unsafe code detected"

        df_safe = st.session_state.df.copy()
        env = {"df": df_safe, "pd": pd, "plt": plt, "sns": sns}

        try:
            exec(cleaned_code, {}, env)

            result = env.get("result", None)

            if isinstance(env.get("df"), pd.DataFrame):
                st.session_state.df = env["df"]

            return result

        except Exception as e:
            return f"Execution Error: {str(e)}"

    # ================= STEP 1: DATA CLEANING =================
    st.markdown("---")
    st.header("🧹 Step 1: Data Cleaning")

    cleaning_prompt = st.text_area(
        "Cleaning Instructions",
        value="Handle missing values, remove duplicates, fix datatypes."
    )

    if st.button("Run Cleaning"):

        with st.spinner("Cleaning data..."):

            prompt = f"""
            You are a pandas expert. Clean dataframe df.

            Rules:
            - Handle missing values
            - Remove duplicates
            - Fix datatypes
            - Modify df directly
            - Do not explain

            User instructions: {cleaning_prompt}
            """

            code = llm.invoke([
                ("system", "Return only Python code."),
                ("human", prompt)
            ]).content

            execute_analytics_code(code)

            st.session_state.cleaning_report = "✅ Data cleaned successfully"
            st.success("Done")

    if st.session_state.cleaning_report:
        st.markdown(st.session_state.cleaning_report)

    # ================= STEP 2: REPORT =================
    st.markdown("---")
    st.header("📊 Step 2: Insights & Charts")

    if st.button("Generate Report"):

        df_plot = st.session_state.df.copy()

        num_cols = df_plot.select_dtypes(include="number").columns
        cat_cols = df_plot.select_dtypes(include="object").columns

        if len(num_cols) > 0:
            plt.figure()
            sns.histplot(df_plot[num_cols[0]].dropna())
            plt.title(num_cols[0])
            plt.savefig("chart1.png")
            plt.close()

        if len(cat_cols) > 0:
            plt.figure()
            df_plot[cat_cols[0]].value_counts().head(10).plot(kind="bar")
            plt.title(cat_cols[0])
            plt.savefig("chart2.png")
            plt.close()

        report = llm.invoke([
            ("system", "You are a business analyst."),
            ("human", f"Columns: {list(df_plot.columns)}, Rows: {len(df_plot)}. Give insights.")
        ])

        st.session_state.insights_report = report.content

    if st.session_state.insights_report:
        st.markdown(st.session_state.insights_report)

        if os.path.exists("chart1.png"):
            st.image("chart1.png")

        if os.path.exists("chart2.png"):
            st.image("chart2.png")

    # ================= STEP 3: CHAT =================
    st.markdown("---")
    st.header("💬 Ask Questions")

    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    def run_chat_agent(user_query: str) -> str:

    system_prompt = """
    You are a Python pandas data analyst.

    STRICT RULES:
    - ONLY generate pandas code
    - NEVER use SQL
    - NEVER use import
    - dataframe is df
    - ALWAYS store output in variable: result
    - DO NOT explain
    - RETURN ONLY CODE

    IMPORTANT:
    - For filtering strings, use:
      df['col'].str.contains(value, case=False, na=False)
    """

    # Step 1 → Generate code
    code_response = llm.invoke([
        ("system", system_prompt),
        ("human", user_query)
    ])

    generated_code = code_response.content.strip()
    generated_code = generated_code.replace("```python", "").replace("```", "")

    # 🔒 Guardrails
    if "import" in generated_code.lower() or "select " in generated_code.lower():
        return "⚠️ I couldn't process that query properly. Try rephrasing your question."

    # Step 2 → Execute
    execution_result = execute_analytics_code(generated_code)

    # Handle execution failure
    if isinstance(execution_result, str) and "Error" in execution_result:
        return f"⚠️ Error while processing your request: {execution_result}"

    # Step 3 → Format result
    if isinstance(execution_result, pd.DataFrame):
        result_str = execution_result.head(30).to_string()
    elif isinstance(execution_result, pd.Series):
        result_str = execution_result.to_string()
    else:
        result_str = str(execution_result)

    # Step 4 → Final explanation
    final_response = llm.invoke([
        ("system", "Explain result clearly in simple terms."),
        ("human", f"""
        User Question: {user_query}

        Result:
        {result_str}
        """)
    ])

    return final_response.content

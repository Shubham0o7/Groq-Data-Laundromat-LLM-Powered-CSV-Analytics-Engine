import streamlit as st
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_core.tools import tool

# ================= LOAD ENV =================
load_dotenv()

# ================= UI =================
st.set_page_config(page_title="LLM CSV Analytics", layout="wide")
st.title("🤖 LLM-Powered CSV Analytics Tool")

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

    if "df" not in st.session_state:
        st.session_state.df = df_raw.copy()
        st.session_state.chat_history = []
        st.session_state.cleaning_done = False
        st.session_state.report_text = ""

    st.subheader("Preview")
    st.dataframe(st.session_state.df.head())

    # ================= EXECUTION ENGINE =================
    def execute_analytics_code(code_input: str):

        cleaned_code = code_input.strip().replace("```python", "").replace("```", "")

        if "import" in cleaned_code or "os." in cleaned_code:
            return "Unsafe code detected"

        df_safe = st.session_state.df.copy()

        sandbox_env = {"df": df_safe, "pd": pd, "plt": plt, "sns": sns}

        try:
            exec(cleaned_code, {}, sandbox_env)
            result = sandbox_env.get("result", None)

            if isinstance(sandbox_env.get("df"), pd.DataFrame):
                st.session_state.df = sandbox_env["df"]

            return result

        except Exception as e:
            return f"Execution Error: {str(e)}"

    # ================= TOOL =================
    @tool
    def python_data_executor(code_input: str) -> str:
        """Run pandas code on df. Must store output in 'result'."""

        result = execute_analytics_code(code_input)

        if isinstance(result, pd.DataFrame):
            return result.head(30).to_string()
        elif isinstance(result, pd.Series):
            return result.to_string()
        else:
            return str(result)

    llm_with_tools = llm.bind_tools([python_data_executor])

    # ================= STEP 1: DATA CLEANING =================
    st.markdown("---")
    st.header("🧹 Step 1: Data Cleaning")

    cleaning_prompt = st.text_area(
        "Cleaning Instructions",
        value="Handle missing values, remove duplicates, fix data types, and clean anomalies."
    )

    if st.button("Run Cleaning"):

        with st.spinner("Cleaning data..."):

            system_prompt = """
            You are a data cleaning expert.

            RULES:
            - Use pandas on DataFrame df
            - Fix missing values
            - Remove duplicates
            - Fix data types
            - Store output in df
            - DO NOT explain
            """

            messages = [
                ("system", system_prompt),
                ("human", cleaning_prompt)
            ]

            ai_msg = llm_with_tools.invoke(messages)

            if ai_msg.tool_calls:
                tool_call = ai_msg.tool_calls[0]
                code = list(tool_call["args"].values())[0]
                python_data_executor.invoke({"code_input": code})

                st.session_state.cleaning_done = True
                st.success("✅ Data cleaned successfully")

    if st.session_state.cleaning_done:
        st.dataframe(st.session_state.df.head())

    # ================= STEP 2: REPORT =================
    st.markdown("---")
    st.header("📊 Step 2: Generate Insights")

    if st.button("Generate Report"):

        df_plot = st.session_state.df.copy()

        numeric_cols = df_plot.select_dtypes(include=['number']).columns
        cat_cols = df_plot.select_dtypes(include=['object']).columns

        # Chart 1
        if len(numeric_cols) > 0:
            col = numeric_cols[0]
            plt.figure()
            sns.histplot(df_plot[col].dropna())
            plt.title(col)
            plt.savefig("chart1.png")
            plt.close()

        # Chart 2
        if len(cat_cols) > 0:
            col = cat_cols[0]
            plt.figure()
            df_plot[col].value_counts().head(10).plot(kind="bar")
            plt.title(col)
            plt.savefig("chart2.png")
            plt.close()

        # LLM report
        report_prompt = f"""
        Dataset columns: {list(df_plot.columns)}
        Rows: {len(df_plot)}

        Give:
        1. Summary
        2. Trends
        3. Recommendations
        """

        report = llm.invoke([
            ("system", "You are a business analyst."),
            ("human", report_prompt)
        ])

        st.session_state.report_text = report.content

    if st.session_state.report_text:
        st.markdown(st.session_state.report_text)

        if os.path.exists("chart1.png"):
            st.image("chart1.png")

        if os.path.exists("chart2.png"):
            st.image("chart2.png")

    # ================= STEP 3: CHAT =================
    st.markdown("---")
    st.header("💬 Step 3: Ask Questions")

    def run_chat_agent(query: str):

        system_prompt = """
        You are a data analyst working with a pandas DataFrame called df.

        RULES:
        - Generate pandas code
        - Store output in variable 'result'
        - Use tool to execute
        - DO NOT return code
        - Return final answer only
        """

        messages = [
            ("system", system_prompt),
            ("human", query)
        ]

        ai_msg = llm_with_tools.invoke(messages)

        if ai_msg.tool_calls:
            tool_call = ai_msg.tool_calls[0]
            code = list(tool_call["args"].values())[0]

            result = python_data_executor.invoke({"code_input": code})

            explanation = llm.invoke([
                ("system", "Explain results clearly."),
                ("human", f"Question: {query}\nResult: {result}")
            ])

            return explanation.content

        return ai_msg.content

    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if user_query := st.chat_input("Ask your data question"):

        with st.chat_message("user"):
            st.markdown(user_query)

        st.session_state.chat_history.append({"role": "user", "content": user_query})

        with st.spinner("Thinking..."):

            response = run_chat_agent(user_query)

            with st.chat_message("assistant"):
                st.markdown(response)

            st.session_state.chat_history.append({"role": "assistant", "content": response})

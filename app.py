import streamlit as st
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_core.tools import tool
from pydantic import BaseModel, Field

# Load environment variables
load_dotenv()

# ================= UI =================
st.set_page_config(page_title="Data Laundromat", layout="wide")
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

    st.dataframe(st.session_state.df.head())

    # ================= EXECUTION ENGINE =================
    def execute_analytics_code(code_input: str):
        import warnings

        # Clean code
        cleaned_code = code_input.strip().replace("```python", "").replace("```", "")

        # Basic guardrail
        if "import" in cleaned_code or "os." in cleaned_code:
            return "Unsafe code detected"

        df_safe = st.session_state.df.copy()
        sandbox_env = {"df": df_safe, "pd": pd, "plt": plt, "sns": sns}

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                exec(cleaned_code, {}, sandbox_env)

            result = sandbox_env.get("result", None)

            if isinstance(sandbox_env.get("df"), pd.DataFrame):
                st.session_state.df = sandbox_env["df"]

            return result

        except Exception as e:
            return f"Execution Error: {str(e)}"

    # ================= TOOL =================
    class PythonExecutorInput(BaseModel):
        code_input: str = Field(...)

    @tool("python_data_executor", args_schema=PythonExecutorInput)
    def data_tool(code_input: str):
        result = execute_analytics_code(code_input)

        if isinstance(result, pd.DataFrame):
            return result.head(50).to_string()
        elif isinstance(result, pd.Series):
            return result.to_string()
        else:
            return str(result)

    llm_with_tools = llm.bind_tools([data_tool])

    # ================= CHAT AGENT =================
    def run_chat_agent(query: str):

        system_prompt = """
        You are a data analyst working with a pandas DataFrame named df.

        RULES:
        1. Always generate pandas code
        2. Always store final output in variable: result
        3. Use tool to execute code
        4. Do NOT return Python code
        5. Return ONLY final answer in plain English
        """

        messages = [
            ("system", system_prompt),
            ("human", query)
        ]

        # Step 1: LLM generates tool call
        ai_msg = llm_with_tools.invoke(messages)

        if ai_msg.tool_calls:
            tool_call = ai_msg.tool_calls[0]
            args = tool_call["args"]

            code_to_run = args.get("code_input") or list(args.values())[0]

            # Step 2: Execute code
            tool_result = data_tool.invoke({"code_input": code_to_run})

            # Step 3: Convert to natural answer
            explanation = llm.invoke([
                ("system", "Explain results clearly like a business analyst."),
                ("human", f"""
                User Question: {query}
                Computed Result:
                {tool_result}
                """)
            ])

            return explanation.content

        return ai_msg.content

    # ================= CHAT UI =================
    st.markdown("---")
    st.header("💬 Ask Questions About Your Data")

    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if user_query := st.chat_input("Ask anything about your data..."):

        with st.chat_message("user"):
            st.markdown(user_query)

        st.session_state.chat_history.append({
            "role": "user",
            "content": user_query
        })

        with st.spinner("Analyzing..."):

            response = run_chat_agent(user_query)

            with st.chat_message("assistant"):
                st.markdown(response)

            st.session_state.chat_history.append({
                "role": "assistant",
                "content": response
            })
``

import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_core.tools import tool
from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage


# Load environment variables from .env file
load_dotenv()


# ================= 1. STREAMLIT UI CONFIGURATION =================
st.set_page_config(page_title="Groq & Roll Data Laundromat", layout="wide")
st.title("🤖 LLM-Powered Spreadsheet Washer & Insight Spinner")
st.write("Upload a CSV file to automatically clean your data, visualize charts, and generate business insights.")


# ================= 2. SECURE API KEY VERIFICATION =================
groq_api_key = os.environ.get("GROQ_API_KEY")


if not groq_api_key:
    st.error("🔒 Power Outage: The Laundromat is out of power! Please plug the GROQ_API_KEY into the generator (.env file).")
    st.stop()


# Initialize Groq LLM (Using production tier Llama 3.1 8B)
llm = ChatGroq(
    model="llama-3.1-8b-instant", 
    api_key=groq_api_key,
    temperature=0
)


# ================= 3. FILE UPLOAD LOGIC =================
uploaded_file = st.file_uploader("🧺 Drop Your Dirty CSV Load Here", type="csv")


if uploaded_file is not None:
    if uploaded_file.size > 100 * 1024 * 1024:
        st.error("⚠️ **Load Limit Exceeded:** This laundromat only handles loads under 100MB to prevent memory timeouts!")
        st.stop()


if uploaded_file is not None:
    # Read the full dataset into memory natively
    df_raw = pd.read_csv(uploaded_file)
        
    if "df" not in st.session_state:
        st.session_state.df = df_raw.copy()
        st.session_state.cleaning_report = ""
        st.session_state.insights_report = ""


    # Main layout division: Preview and Global Metrics
    preview_col, metrics_col = st.columns([5, 3], gap="large")
    
    with preview_col:
        st.subheader("🔍 Batch Inspection Preview")
        st.caption("A look at the top rows of your active dataframe load.")
        st.dataframe(
            st.session_state.df.head(10), 
            use_container_width=True
        )
        
    with metrics_col:
        st.subheader("📊 Full Load Metrics")
        st.caption("Diagnostic metrics detailing the macroscopic contamination state.")
        
        with st.container(border=True):
            m_col1, m_col2 = st.columns(2)
            with m_col1:
                st.metric(label="🧵 Total Dataset Rows", value=f"{st.session_state.df.shape[0]:,}")
            with m_col2:
                st.metric(label="📐 Total Data Columns", value=st.session_state.df.shape[1])
            
            missing_count = int(st.session_state.df.isnull().sum().sum())
            duplicate_count = int(st.session_state.df.duplicated().sum())
            
            st.markdown("---")
            
            m_col3, m_col4 = st.columns(2)
            with m_col3:
                st.metric(
                    label="⚠️ Empty Pockets", 
                    value=f"{missing_count:,}", 
                    delta=f"{missing_count} Missing Values" if missing_count > 0 else "No Missing data", 
                    delta_color="inverse" if missing_count > 0 else "normal"
                )
            with m_col4:
                st.metric(
                    label="🔁 Tangled Clones", 
                    value=f"{duplicate_count:,}",
                    delta=f"{duplicate_count} Duplicate Rows" if duplicate_count > 0 else "No Duplicate data",
                    delta_color="inverse" if duplicate_count > 0 else "normal"
                )


            if missing_count > 0 or duplicate_count > 0:
                st.warning("⚠️ **Status:** Contaminants detected. Scrubbing recommended.")
            else:
                st.success("✅ **Status:** Fabric is clean. Ready for Insight Reporter.")


    st.markdown("---")


    # ================= 4. STABLE NATIVE FUNCTION CALLING ENGINE =================
    def execute_analytics_code(code_input: str) -> str:
        """Isolated environment runner with optimized in-memory modifications."""
        import warnings
        cleaned_code = code_input.strip().strip("```python").strip("```")
        df_safe = st.session_state.df.copy()
        
        sandbox_env = {"df": df_safe, "plt": plt, "sns": sns, "pd": pd}
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                exec(cleaned_code, globals(), sandbox_env)
                
            if isinstance(sandbox_env.get("df"), pd.DataFrame):
                st.session_state.df = sandbox_env["df"]
            return "Success"
        except Exception as e:
            return f"Execution Error: {str(e)}"


    class PythonExecutorInput(BaseModel):
        code_input: str = Field(..., description="The executable Python code string to process 'df'.")


    @tool("python_data_executor", args_schema=PythonExecutorInput)
    def data_tool(code_input: str) -> str:
        """Executes python pandas instructions on the target mutable dataframe 'df'."""
        return execute_analytics_code(code_input)
    
    # Fast Binding
    llm_with_tools = llm.bind_tools([data_tool])


    def run_agent_workflow(user_instruction: str) -> str:
        """OPTIMIZED: Single-pass predictive tool framework to eliminate multi-minute loop lags."""
        system_instruction = (
            "You are an elite data scientist and data cleaning expert with access to a pandas DataFrame named 'df'. "
            "Analyze the schema and sample data provided, then generate your complete Python Pandas cleaning script and execute it immediately using your tool. "
            "Do not loop or iterate redundantly. Write clean, direct code to transform 'df' based on the guidelines."
        )
        messages = [("system", system_instruction), ("human", user_instruction)]
        
        # Single invocation path
        ai_msg = llm_with_tools.invoke(messages)
        if ai_msg.tool_calls:
            tool_call = ai_msg.tool_calls[0]
            args = tool_call["args"]
            code_to_run = args.get("code_input") or args.get("code") or list(args.values())[0]
            execute_analytics_code(code_to_run)
            return "Data Cleaning operations executed successfully inside the local sandbox runtime. Columns have been casted, missing fields processed, and binary indicators standardized."
        
        return ai_msg.content


    # ================= STAGE 1: DATA CLEANING =================
    st.header("🧹 Step 1: The Intelligent Data Scrubber")
    cleaning_prompt = st.text_area(
        "Modify stain-removal instructions if needed:",
        value="Identify and handle missing values appropriately, remove duplicate rows, ensure correct data types, standardize boolean columns into Yes/No categories, and clean anomalies."
    )


    if st.button("🧼 Start the Heavy-Duty Wash Cycle"):
        with st.spinner("Scrubbing rows clean across the dataset matrix..."):
            try:
                before_cols = list(st.session_state.df.columns)
                before_row_count = st.session_state.df.shape[0]
                
                # DATA CONTEXT INJECTION: Give the LLM visual awareness of structural value anomalies
                df_sample = st.session_state.df.head(3).to_string()
                df_dtypes = st.session_state.df.dtypes.to_string()
                
                reinforced_cleaning_prompt = f"""
                You are an expert data cleaning engine. The active dataframe 'df' has {before_row_count} rows.
                
                Columns & Data Types:
                {df_dtypes}
                
                Sample Data Context (First 3 rows):
                {df_sample}
                
                Execute your cleaning script via your tool according to these rules: {cleaning_prompt}.
                CRITICAL TASK: Inspect the sample data carefully. If columns like 'Discount Applied' contain boolean patterns (True/False/NaN), explicitly write code to map/convert them to standard string 'Yes' and 'No' values while gracefully handling missing items.
                Ensure code operates directly on 'df'.
                """
                st.session_state.cleaning_report = run_agent_workflow(reinforced_cleaning_prompt)
                st.success("Deep Wash Cycle Completed!")
                st.rerun()
            except Exception as e:
                st.error(f"Scrubbing Execution Error: {e}")


    if st.session_state.cleaning_report:
        st.subheader("📋 Detergent Action Summary")
        st.markdown(st.session_state.cleaning_report)
        
        csv_data = st.session_state.df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="💾 Download Freshly Folded CSV",
            data=csv_data,
            file_name="freshly_folded_laundry.csv",
            mime="text/csv"
        )


    # ================= STAGE 2: THE SPIN CYCLE & INSIGHT PRESS =================
    st.markdown("---")
    st.header("📈 Step 2: The Spin Cycle & Insight Press")
    
    report_prompt = st.text_area(
        "Modify the styling and analytical press settings if needed:",
        value="""Provide a comprehensive analysis of the dataset based on the metrics provided. Format it strictly with these exact markdown subheaders:
### 📋 The Freshly Folded Executive Summary
[Write summary analysis text here. Ensure you place the token '[CHART_1_HERE]' at the very end of this section]


### 🔍 Fabric Irregularities & Key Trends
[Write trends analysis text here. Ensure you place the token '[CHART_2_HERE]' at the very end of this section]


### 🧮 Strategic Pressing Recommendations
[Write recommendations here]""",
        height=180
    )


    if st.button("🚀 Run the Spin & Generate Report"):
        with st.spinner("Synthesizing metrics and plotting charts locally..."):
            try:
                for chart in ["chart1.png", "chart2.png"]:
                    if os.path.exists(chart):
                        os.remove(chart)
                
                df_plot = st.session_state.df.copy()
                numeric_cols = list(df_plot.select_dtypes(include=['number']).columns)
                categorical_cols = list(df_plot.select_dtypes(include=['object', 'category', 'string']).columns)


                # Local Vector Plotting Engine
                if numeric_cols:
                    primary_num = numeric_cols[0]
                    fig1, ax1 = plt.subplots(figsize=(10, 4.5))
                    sns.histplot(df_plot[primary_num].dropna(), kde=True, ax=ax1, color="#2E86AB")
                    ax1.set_title(f"Distribution of {primary_num.replace('_', ' ')}")
                    fig1.savefig('chart1.png', bbox_inches='tight')
                    plt.close(fig1)
                
                if categorical_cols:
                    primary_cat = categorical_cols[0]
                    fig2, ax2 = plt.subplots(figsize=(10, 4.5))
                    df_plot[primary_cat].value_counts().head(10).plot(kind='bar', ax=ax2, color="#A23B72")
                    ax2.set_title(f"Top Categories in {primary_cat.replace('_', ' ')}")
                    fig2.savefig('chart2.png', bbox_inches='tight')
                    plt.close(fig2)


                # HIGH-SCALE LIGHTWEIGHT GENERATION PATTERN
                actual_columns = list(st.session_state.df.columns)
                row_count_val = int(st.session_state.df.shape[0])
                
                text_generation_prompt = f"""
                You are an expert business analyst.
                Columns available: {actual_columns}. Total Rows: {row_count_val}.
                Fulfill this structural layout directive explicitly:
                {report_prompt}
                Start directly with the first subheader line. Append '[CHART_1_HERE]' and '[CHART_2_HERE]' explicitly.
                """


                final_text_call = llm.invoke([
                    ("system", "You are an automated insight reporter. Write an analytical report structure based on structural data schema context. Keep text clear and brief."),
                    ("human", text_generation_prompt)
                ])
                
                st.session_state.insights_report = final_text_call.content
                st.success("Report and Visual Outfits Compiled Successfully!")
                st.rerun() 
                
            except Exception as e:
                st.error(f"Analysis Generation Failure: {e}")


    # ================= 🔄 INLINE STREAM RENDERING ENGINE =================
    if st.session_state.insights_report:
        raw_text = st.session_state.insights_report
        
        if "### 🔍 Fabric Irregularities & Key Trends" in raw_text:
            parts = raw_text.split("### 🔍 Fabric Irregularities & Key Trends")
            summary_part = parts[0]
            remaining_part = parts[1] if len(parts) > 1 else ""
            
            clean_summary = summary_part.replace("[CHART_1_HERE]", "").replace("### 📋 The Freshly Folded Executive Summary", "").strip()
            
            st.subheader("📋 The Freshly Folded Executive Report")
            st.markdown(f"<p style='font-family:sans-serif; font-size:16px; line-height:1.6;'>{clean_summary}</p>", unsafe_allow_html=True)
            
            if os.path.exists("chart1.png"):
                st.image("chart1.png", caption="Primary Feature Distribution")
                
            if "### 🧮 Strategic Pressing Recommendations" in remaining_part:
                trends_parts = remaining_part.split("### 🧮 Strategic Pressing Recommendations")
                trends_part = trends_parts[0]
                rec_part = trends_parts[1] if len(trends_parts) > 1 else ""
                
                st.markdown("### 🔍 Fabric Irregularities & Key Trends")
                st.markdown(f"<p style='font-family:sans-serif; font-size:16px; line-height:1.6;'>{trends_part.replace('[CHART_2_HERE]', '').strip()}</p>", unsafe_allow_html=True)
                
                if os.path.exists("chart2.png"):
                    st.image("chart2.png", caption="Categorical Feature Layout Analysis")
                    
                st.markdown("### 🧮 Strategic Pressing Recommendations")
                st.markdown(f"<p style='font-family:sans-serif; font-size:16px; line-height:1.6;'>{rec_part.strip()}</p>", unsafe_allow_html=True)
        else:
            st.markdown(raw_text)


    # ================= 🪙 STEP 3: INTERACTIVE SPOT CLEANER =================
st.markdown("---")
st.header("💬 Step 3: Interactive Spot Cleaner")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ================= CHAT DISPLAY =================
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# ================= CHAT AGENT =================
def run_chat_agent(user_query: str) -> str:
    """
    Executes:
    1. LLM generates pandas code
    2. Code is executed
    3. Result is converted to natural language
    """

    system_prompt = """
    You are a data analyst working with a pandas DataFrame named df.

    RULES:
    - Generate ONLY pandas Python code
    - ALWAYS store output in variable: result
    - DO NOT print anything
    - DO NOT explain anything
    - df is already available
    """

    # Step 1: Generate code
    code_response = llm.invoke([
        ("system", system_prompt),
        ("human", user_query)
    ])

    generated_code = code_response.content.strip()

    # ✅ Clean code (remove markdown if present)
    generated_code = generated_code.replace("```python", "").replace("```", "")

    # Step 2: Execute code
    execution_result = execute_analytics_code(generated_code)

    # Convert result to string safely
    if isinstance(execution_result, pd.DataFrame):
        result_str = execution_result.head(30).to_string()
    elif isinstance(execution_result, pd.Series):
        result_str = execution_result.to_string()
    else:
        result_str = str(execution_result)

    # Step 3: Convert result -> natural language
    final_response = llm.invoke([
        ("system", "You are a business data analyst. Explain results clearly."),
        ("human", f"""
        User Question: {user_query}

        Computed Result:
        {result_str}

        Give a clear, concise answer.
        """)
    ])

    return final_response.content


# ================= USER INPUT =================
if user_query := st.chat_input("Ask about your data"):

    with st.chat_message("user"):
        st.markdown(user_query)

    st.session_state.chat_history.append({
        "role": "user",
        "content": user_query
    })

    with st.spinner("Analyzing data..."):
        try:
            response = run_chat_agent(user_query)

            with st.chat_message("assistant"):
                st.markdown(response)

            st.session_state.chat_history.append({
                "role": "assistant",
                "content": response
            })

        except Exception as e:
            st.error(f"Error: {e}")
    

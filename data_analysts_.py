from langchain_core import language_models
import re
import streamlit as st
from langchain_experimental.tools import PythonREPLTool
from langchain_core.tools import tool
from ddgs import DDGS
from langgraph.prebuilt import create_react_agent
from langgraph_supervisor import create_supervisor

from langchain_google_genai import ChatGoogleGenerativeAI

# from groq import Groq
from dotenv import load_dotenv
import os

load_dotenv()
model = ChatGoogleGenerativeAI(model="gemini-3.5-flash", api_key=os.getenv("GEMINI_API_KEY"))

@tool
def search_web(query: str) -> str:
    """Search the web for up-to-date information, references, and articles."""
    try:
        results = list(DDGS().text(query, max_results=3))
        if not results:
            return "No web results found for this query. Please synthesize using your foundational knowledge."
        formatted = []
        for r in results:
            formatted.append(f"Title: {r.get('title')}\nLink: {r.get('href')}\nSnippet: {r.get('body')}")
        return "\n\n".join(formatted)
    except Exception as e:
        return f"Web search temporarily returned an error ({e}). Please synthesize your answer using your foundational knowledge."

@tool
def check_file_exists(file_path: str) -> str:
    """Checks whether a file exists on disk.
    Use this to verify if cleaned or processed datasets already exist before running code.
    """
    if os.path.exists(file_path):
        return f"File '{file_path}' EXISTS and is ready for use."
    return f"File '{file_path}' DOES NOT exist."

query_refiner_prompt = (
    "You are a query refiner agent.\n\n"
    "INSTRUCTIONS:\n"
    "- Refine the user's query to make it more specific and actionable\n"
    "- Respond ONLY with the refined query, do NOT include any other text."
)

data_cleaning_prompt = (
    "You are an expert Data Cleaning and Preparation Agent.\n\n"
    "OBJECTIVE:\n"
    "Inspect, sanitize, and validate the dataset so it is robust for statistical analysis.\n\n"
    "INSTRUCTIONS:\n"
    "1. Check Existing Cache FIRST:\n"
    "   - Use `check_file_exists` on the expected target output (e.g., 'cleaned_student_data.csv' or 'cleaned_data.csv').\n"
    "   - If the file ALREADY EXISTS, DO NOT re-run cleaning code. Immediately return a report confirming the cleaned file is ready, along with its path.\n"
    "2. If the file does NOT exist:\n"
    "   - Load the raw data with python_repl_tool.\n"
    "   - Standardize column names, handle missing values/duplicates, cast correct types.\n"
    "   - Save the result to 'cleaned_student_data.csv'.\n\n"
    "OUTPUT CONTRACT:\n"
    "Provide a concise summary:\n"
    "- Status: (Existing file reused / New file cleaned)\n"
    "- File path of the cleaned dataset"
)


eda_prompt = (
    "You are an expert Exploratory Data Analysis (EDA) Agent.\n\n"
    "OBJECTIVE:\n"
    "Analyze the cleaned dataset to uncover distributions, statistical summaries, correlations, and notable anomalies.\n\n"
    "INSTRUCTIONS:\n"
    "1. Statistical Profiling: Compute five-number summaries (min, Q1, median, Q3, max), mean, standard deviation, and skewness for numeric fields.\n"
    "2. Categorical Profiling: Determine unique value counts, top categories, and frequency distributions for categorical fields.\n"
    "3. Correlation & Relationships: Calculate correlation matrices (Pearson/Spearman) to spot multicollinearity and strong dependencies.\n"
    "4. Hypothesis Formulation: Formulate 3-5 specific, testable business hypotheses and data patterns worth visualizing in the next phase.\n"
    "5. Do NOT generate final presentation plots here; focus on data profiling and statistical discovery.\n\n"
    "OUTPUT CONTRACT:\n"
    "Provide a structured EDA Report containing:\n"
    "- Key descriptive statistics table/summary\n"
    "- Top 3 strongest feature correlations or trends discovered\n"
    "- 3-5 clear hypotheses/questions to be explored by the Analysis & Visualization Agent"
)

analysis_viz_prompt = (
    "You are an expert Data Analysis & Visualization Agent.\n\n"
    "OBJECTIVE:\n"
    "Answer the core user goals and test EDA hypotheses by generating high-impact visualizations and actionable business insights.\n\n"
    "INSTRUCTIONS:\n"
    "1. Hypothesis Testing: Evaluate the hypotheses and key trends identified in the EDA phase using the cleaned dataset.\n"
    "2. Visualization Standards:\n"
    "   - Create publication-quality charts using Matplotlib, Seaborn, or Plotly.\n"
    "   - Every plot MUST have a clear title, labeled axes with units, informative legends, and a clean aesthetic.\n"
    "   - Save all plots as PNG files to an output directory (e.g., 'visualizations/figure_X.png').\n"
    "3. Synthesis: Interpret each chart directly—explain what the trend or distribution means in business context.\n"
    "4. Strategic Recommendations: Provide actionable takeaways and concrete next steps based on the empirical evidence.\n\n"
    "OUTPUT CONTRACT:\n"
    "Provide the Final Analytics Brief containing:\n"
    "- Executive Summary addressing the primary question\n"
    "- List of generated chart files with captions and key takeaways\n"
    "- Data-driven conclusions and recommended decisions"
)


supervisor_prompt = (
    "You are the Lead Data Science Supervisor.\n\n"
    "INSTRUCTIONS:\n"
    "1. Orchestrate the pipeline in strict logical order:\n"
    "   Step 1: Delegate to Data Cleaning Agent to sanitize and validate the data.\n"
    "   Step 2: Delegate to EDA Agent to profile the cleaned data and form hypotheses.\n"
    "   Step 3: Delegate to Analysis & Visualization Agent to produce final charts and insights.\n"
    "2. Ensure each agent receives the necessary context (like file paths and previous findings) from earlier steps.\n"
    "3. Once the Analysis & Visualization agent finishes, synthesize the complete final response for the user and terminate the workflow."
)

# Create the tool instance
python_repl_tool = PythonREPLTool()


query_refiner_agent = create_react_agent(
    model=model, 
    tools=[],
    prompt=query_refiner_prompt,
    name="query_refiner_agent"
)

cleaner_agent = create_react_agent(
    model=model,
    tools=[check_file_exists, python_repl_tool], # e.g. python code execution / file tools
    prompt=data_cleaning_prompt,
    name="data_cleaning_agent"
)

eda_agent = create_react_agent(
    model=model,
    tools=[python_repl_tool],
    prompt=eda_prompt,
    name="eda_agent"
)

viz_agent = create_react_agent(
    model=model,
    tools=[python_repl_tool],
    prompt=analysis_viz_prompt,
    name="analysis_viz_agent"
)


# Create the supervisor agent
supervisor_agent = create_supervisor(
    model=model,
    agents=[query_refiner_agent, cleaner_agent, eda_agent, viz_agent],
    prompt=supervisor_prompt
)

app = supervisor_agent.compile()


print("\nStarting multi-agent researcher...\n")

inputs = {
    "messages": [
        {
            "role": "user",
            "content": (
                "Please analyze the dataset located at 'student_data_test.csv'.\n"
                "Objective: Clean the data, perform an in-depth EDA to discover key student performance factors, "
                "and generate visualizations with strategic insights."
            ),
        }
    ]
}


if __name__ == "__main__":
    print("\n" + "=" * 65, flush=True)
    print("🚀 STARTING MULTI-AGENT DATA ANALYST PIPELINE", flush=True)
    print("=" * 65 + "\n", flush=True)
    print("⏳ Contacting supervisor to orchestrate the pipeline...\n", flush=True)

    # Use stream_mode="updates" to get notified as each agent node executes
    for chunk in app.stream(inputs, stream_mode="updates"):
        for node_name, update in chunk.items():
            print(f"\n{'='*20} 🤖 AGENT CALLED: [{node_name.upper()}] {'='*20}", flush=True)
            
            node_messages = update.get("messages", []) if isinstance(update, dict) else []
            for msg in node_messages:
                # Check if supervisor is delegating to a sub-agent
                if hasattr(msg, "tool_calls") and msg.tool_calls:
                    for tc in msg.tool_calls:
                        target_agent = tc.get("name", "unknown")
                        print(f"👉 Delegating to sub-agent / tool: [{target_agent}]", flush=True)
                
                # Pretty-print the agent's thought process and outputs
                if hasattr(msg, "pretty_print"):
                    msg.pretty_print()
                    print(flush=True)
                elif hasattr(msg, "content"):
                    print(f"[{node_name}]: {msg.content}\n", flush=True)

# Query refiner
# cleaning data
# EDA
# Analysis and Visualizations

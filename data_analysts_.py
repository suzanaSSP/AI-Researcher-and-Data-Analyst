from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated, Optional
from langchain_core.messages import BaseMessage, AIMessage
from langgraph.graph.message import add_messages
import re
import streamlit as st
from langchain_experimental.tools import PythonREPLTool
from langchain_core.tools import tool
from ddgs import DDGS
from langgraph.prebuilt import create_react_agent
from langgraph_supervisor import create_supervisor

from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os

load_dotenv()

# Connect to LangSmith for full tracing of all agent steps and tools
os.environ["LANGCHAIN_TRACING_V2"] = "true"
if os.getenv("LANGSMITH_API_KEY") and not os.getenv("LANGCHAIN_API_KEY"):
    os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGSMITH_API_KEY")
os.environ.setdefault("LANGCHAIN_PROJECT", "ai-data-analyst")

# Groq-hosted Qwen model (max_tokens set to 1000 per Groq OTPM rate limits)
model = ChatGroq(
    model_name="qwen/qwen3.8-27b",
    api_key=os.getenv("GROQ_API_KEY"),
    max_tokens=1000
)

class PipelineState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    cleaned_data_path: Optional[str]
    eda_code: Optional[list[str]]             
    viz_code: Optional[list[str]]             
    visualizations_dir: Optional[str]
    visualization_files: Optional[list[str]]
    report: Optional[str]
    task: Optional[str]



# @tool
# def search_web(query: str) -> str:
#     """Search the web for up-to-date information, references, and articles."""
#     try:
#         results = list(DDGS().text(query, max_results=3))
#         if not results:
#             return "No web results found for this query. Please synthesize using your foundational knowledge."
#         formatted = []
#         for r in results:
#             formatted.append(f"Title: {r.get('title')}\nLink: {r.get('href')}\nSnippet: {r.get('body')}")
#         return "\n\n".join(formatted)
#     except Exception as e:
#         return f"Web search temporarily returned an error ({e}). Please synthesize your answer using your foundational knowledge."

@tool
def check_file_exists(file_path: str) -> str:
    """Checks whether a file exists on disk.
    Use this to verify if cleaned or processed datasets already exist before running code.
    """
    if os.path.exists(file_path):
        return f"File '{file_path}' EXISTS and is ready for use."
    return f"File '{file_path}' DOES NOT exist."

def extract_executed_code(messages: list[BaseMessage]) -> list[str]:
    """Extracts all Python code snippets passed to python_repl_tool."""
    code_snippets = []
    for msg in messages:
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            for tc in msg.tool_calls:
                # Catches both 'query' and 'code' argument names
                if "python" in tc.get("name", "").lower() or "repl" in tc.get("name", "").lower():
                    code = tc.get("args", {}).get("query") or tc.get("args", {}).get("code") or ""
                    if code:
                        code_snippets.append(code.strip())
    return code_snippets



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
    tools=[check_file_exists, python_repl_tool],
    prompt=eda_prompt,
    name="eda_agent"
)

viz_agent = create_react_agent(
    model=model,
    tools=[python_repl_tool],
    prompt=analysis_viz_prompt,
    name="analysis_viz_agent"
)

def query_refiner_node(state: PipelineState) -> dict:
    print("\n" + "=" * 55, flush=True)
    print("🔍 [TASK STARTED]: Query Refinement", flush=True)
    print("=" * 55, flush=True)
    return query_refiner_agent.invoke(state)

def cleaner_node(state: PipelineState) -> dict:
    print("\n" + "=" * 55, flush=True)
    print("🧹 [TASK STARTED]: Data Cleaning & Sanitation", flush=True)
    print("=" * 55, flush=True)
    return cleaner_agent.invoke(state)

def eda_node(state: PipelineState) -> dict:
    print("\n" + "=" * 55, flush=True)
    print("🔬 [TASK STARTED]: Exploratory Data Analysis (EDA)", flush=True)
    print("=" * 55, flush=True)
    initial_count = len(state.get("messages", []))
    result = eda_agent.invoke(state)
    
    # Extract only the code executed during THIS agent's turn
    new_messages = result["messages"][initial_count:]
    code_list = extract_executed_code(new_messages)
    
    print(f"\n📝 [EDA COMPLETED]: Captured {len(code_list)} executed code snippet(s).", flush=True)
    return {
        "messages": result["messages"],
        "eda_code": code_list
    }

def viz_node(state: PipelineState) -> dict:
    print("\n" + "=" * 55, flush=True)
    print("📊 [TASK STARTED]: Analysis & Visualizations", flush=True)
    print("=" * 55, flush=True)
    initial_count = len(state.get("messages", []))
    result = viz_agent.invoke(state)
    
    new_messages = result["messages"][initial_count:]
    code_list = extract_executed_code(new_messages)
    
    # Optional: collect generated image paths
    viz_dir = state.get("visualizations_dir") or "visualizations"
    chart_files = []
    if os.path.exists(viz_dir):
        chart_files = [os.path.join(viz_dir, f) for f in os.listdir(viz_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.svg'))]
        
    print(f"\n📈 [VIZ COMPLETED]: Captured {len(code_list)} code snippet(s) and {len(chart_files)} chart(s).", flush=True)
    return {
        "messages": result["messages"],
        "viz_code": code_list,
        "visualization_files": chart_files
    }


def check_cleaned_data_exists(state: PipelineState) -> str:
    """Checks whether the cleaned dataset exists on disk before letting EDA proceed."""
    cleaned_path = state.get("cleaned_data_path") or "cleaned_student_data.csv"
    if os.path.exists(cleaned_path):
        print(f"\n✅ Cleaned data found at '{cleaned_path}'. Proceeding to EDA...", flush=True)
        return "eda_agent"
    print(f"\n❌ Cleaned data NOT found at '{cleaned_path}'. Going to cleaner agent", flush=True)
    return "cleaner_agent"


builder = StateGraph(PipelineState)
builder.add_node('query_refiner', query_refiner_node)
builder.add_node('cleaner_agent', cleaner_node)
builder.add_node('eda_agent', eda_node)
builder.add_node('viz_agent', viz_node)

builder.add_edge(START, 'query_refiner')

# After query refinement, check if cleaned data already exists:
# - If YES -> skip straight to eda_agent
# - If NO  -> route to cleaner_agent
builder.add_conditional_edges(
    'query_refiner',
    check_cleaned_data_exists,
    {
        'eda_agent': 'eda_agent',
        'cleaner_agent': 'cleaner_agent'
    }
)

# If cleaner runs, proceed to eda_agent next
builder.add_edge('cleaner_agent', 'eda_agent')
builder.add_edge('eda_agent', 'viz_agent')
builder.add_edge('viz_agent', END)

graph = builder.compile()

png_bytes = graph.get_graph().draw_mermaid_png()
with open("pipeline_graph.png", "wb") as f:
    f.write(png_bytes)

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
    ],
    "cleaned_data_path": "cleaned_student_data.csv",
    "visualizations_dir": "visualizations"
}

if __name__ == "__main__":
    print("\n" + "=" * 65, flush=True)
    print("🚀 STARTING STATE GRAPH DATA ANALYST PIPELINE", flush=True)
    print("=" * 65 + "\n", flush=True)

    final_state = {}
    # Use stream_mode="updates" to get notified as each agent node executes
    for chunk in graph.stream(inputs, stream_mode="updates"):
        for node_name, update in chunk.items():
            if isinstance(update, dict):
                final_state.update(update)
            print(f"\n{'='*20} 🤖 NODE FINISHED: [{node_name.upper()}] {'='*20}", flush=True)
            
            node_messages = update.get("messages", []) if isinstance(update, dict) else []
            for msg in node_messages:
                # Check for tool calls
                if hasattr(msg, "tool_calls") and msg.tool_calls:
                    for tc in msg.tool_calls:
                        target_tool = tc.get("name", "unknown")
                        args = tc.get("args", {})
                        code_preview = args.get("query") or args.get("code") or args.get("file_path") or ""
                        print(f"👉 [TOOL CALL]: {target_tool}", flush=True)
                        if code_preview:
                            first_line = str(code_preview).strip().split("\n")[0][:80]
                            print(f"   Executing: {first_line}...", flush=True)
                
                # Pretty-print the message
                if hasattr(msg, "pretty_print"):
                    msg.pretty_print()
                    print(flush=True)
                elif hasattr(msg, "content"):
                    print(f"[{node_name}]: {msg.content}\n", flush=True)

    print("\n" + "=" * 65, flush=True)
    print("🎉 PIPELINE EXECUTION COMPLETED!", flush=True)
    print("=" * 65, flush=True)
    if final_state.get("eda_code"):
        print(f"• EDA Code Snippets Stored: {len(final_state['eda_code'])}", flush=True)
    if final_state.get("viz_code"):
        print(f"• Viz Code Snippets Stored: {len(final_state['viz_code'])}", flush=True)
    if final_state.get("visualization_files"):
        print(f"• Generated Visualizations ({len(final_state['visualization_files'])}):", flush=True)
        for cf in final_state["visualization_files"]:
            print(f"   - {cf}", flush=True)
    print("=" * 65 + "\n", flush=True)

# Query refiner
# cleaning data
# EDA
# Analysis and Visualizations

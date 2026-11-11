from langchain_core import language_models
import re
import streamlit as st

from langchain_core.tools import tool
from ddgs import DDGS
from langgraph.prebuilt import create_react_agent
from langgraph_supervisor import create_supervisor

from langchain_google_genai import ChatGoogleGenerativeAI

# from groq import Groq
from dotenv import load_dotenv
import os

load_dotenv()
model = ChatGoogleGenerativeAI(model="gemini-3.7-flash", api_key=os.getenv("GEMINI_API_KEY"))

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

query_refiner_prompt = (
    "You are a query refiner agent.\n\n"
    "INSTRUCTIONS:\n"
    "- Refine the user's query to make it more specific and actionable\n"
    "- Respond ONLY with the refined query, do NOT include any other text."
)
research_prompt = (
    "You are a research agent.\n\n"
    "INSTRUCTIONS:\n"
    "- Assist ONLY with research-related tasks, DO NOT do anything else\n"
    "- After you're done with your research, respond to the supervisor directly\n"
    "- Include references (links, titles, etc.) to your findings"
    "- Respond ONLY with the summary of the results, do NOT include ANY other text."
)
supervisor_prompt = (
    "You are a supervisor agent.\n\n"
    "INSTRUCTIONS:\n"
    "- Manage the workflow between the query refiner and researcher agents\n"
    "- Use the Query Refiner Agent to correct typos and enhance the query for better search relevance\n"
    "- Use the Research Agent to retrieve up-to-date information from the web\n"
    "- Respond ONLY with the final response, do NOT include any other text."
)


query_refiner_agent = create_react_agent(
    model=model, 
    tools=[],
    prompt=query_refiner_prompt,
    name="query_refiner_agent"
)

researcher_agent = create_react_agent(
    model=model, 
    tools=[search_web],
    prompt=research_prompt,
    name="research_agent"
)

supervisor_agent = create_supervisor(
    model=model,
    agents=[query_refiner_agent, researcher_agent],
    prompt=supervisor_prompt
)

app = supervisor_agent.compile()

print("\nStarting multi-agent researcher...\n")

inputs = {
    "messages": [
        {
            "role": "user",
            "content": "What is quantom physics?"
        }
    ]
}

num_printed = 0
final_messages = []
for chunk in app.stream(inputs, stream_mode="values"):
    messages = chunk.get("messages", [])
    final_messages = messages
    # Print only new messages as they stream in
    while num_printed < len(messages):
        messages[num_printed].pretty_print()
        print()
        num_printed += 1

if final_messages:
    last_msg = final_messages[-1]
    content = last_msg.content
    if isinstance(content, list):
        research_result = "".join(item.get("text", "") for item in content if isinstance(item, dict))
    else:
        research_result = content
    print("\n--- Final Research Result ---")
    print(research_result)


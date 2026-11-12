from crewai import Crew, Task, Agent, Process, LLM
from crewai_tools import SerperDevTool
from langchain_ibm import WatsonxLLM
import os
import getpass
from dotenv import load_dotenv

load_dotenv()

os.environ["GROQ_API_KEY"] = getpass.getpass()

# Parameters
parameters = {"decoding_method": "greedy", "max_new_tokens": 500}

# Create the first LLM
groq_llm = LLM(
    model="groq/llama-3.3-70b-versatile",  # Or "groq/llama-3.1-8b-instant"
    temperature=0.5,
)

# Tools
search = SerperDevTool()

# Create the agent
researcher = Agent(
    role="Senior Research Analyst",
    goal="Uncover cutting-edge developments in AI",
    backstory="You are an expert researcher with a keen eye for details.",
    verbose=True,
    llm=groq_llm,
)

# Create a task
task1 = Task(
    description="Search the internet and find 5 examples of promising AI research.",
    expected_output="A detailed bullet point summary on each of the topics. Each bullet point should cover the topic, background and why the innovation is useful.",
    output_file="task1output.txt",
    agent=researcher,
)

# Create the second agent
writer = Agent(
    role="Content Strategist",
    goal="Craft compelling summaries from research findings",
    backstory="You synthesize complex findings into clear narratives.",
    verbose=True,
    llm=groq_llm,
)

# Create a task
task2 = Task(
    description="Write an engaging keynote speech on quantum computing.",
    expected_output="A detailed keynote speech with an intro, body and conclusion.",
    output_file="task2output.txt",
    agent=writer,
)

# Put all together with the crew
crew = Crew(
    agents=[researcher, writer],
    tasks=[task1, task2],
    process=Process.sequential,
    max_rpm=20  # Keeps requests within Groq free-tier rate limits
)

print(crew.kickoff())
import os
from unittest import result
from dotenv import load_dotenv
from dynamiq.nodes.llms.apenai import OpenAI
from dynamiq.connections import OpenAI as OpenAIConnection, E2B as E2BConnection
from dynamiq.nodes.agents.react import ReActAgent
from dynamiq.nodes.tools.e2b_sandbox import E2bInterpreterTool

#load environment variables from .env file
load_dotenv()

#setup a tool for the agent to use
run_code_tool = E2bInterpreterTool(
    connection=E2BConnection(api_key=os.getenv("E2B_API_KEY")))

#setup a llm
llm = OpenAI(
    connection=OpenAIConnection(api_key=os.getenv("OPENAI_API_KEY")),
    model="gpt-4o",
    temperature=0.2,
    max_tokens=1000)

#create agent with the llm and tools
agent = ReActAgent(
    name="react-agent",
    llm=llm,
    tools=[run_code_tool],
    role="Senior Data Scientist",
    description="An agent that can write and run code to answer questions about data science and machine learning. It can use the E2B tool to run code and get results.",
    max_loops=10
)

#run the agent with a prompt
result = agent.run("Write a python function to calculate the factorial of a number and run it for the number 5.")
print(result.output.get("content"))
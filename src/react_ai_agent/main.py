import os
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
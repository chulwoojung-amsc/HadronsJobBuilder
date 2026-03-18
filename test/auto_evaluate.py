from langchain_openai import ChatOpenAI
from hadrons_config import *
from langchain.messages import (
    SystemMessage,
    HumanMessage,
    ToolCall,
    AIMessage
)

llm = ChatOpenAI(
    model="gpt-oss:120b",
    openai_api_key="sk-local",
    openai_api_base="http://localhost:11434/v1",
    temperature=0
)

amsc_llm_0t = ChatOpenAI(
    model="gpt-oss-120b",
    base_url="https://api.i2-core.american-science-cloud.org/",
    temperature=0
)

#llm = ChatOpenAI(
#    model="gpt-oss-120b-GGUF",
#    openai_api_key="sk-local",
#    openai_api_base="http://localhost:8000/v1"
#)

zerot_llm = ChatOpenAI(
    model="gpt-oss-120b-GGUF",
    openai_api_key="sk-local",
    openai_api_base="http://localhost:8000/v1",
    temperature=0
)

quadro_llm = ChatOpenAI(
    model="gpt-oss-120b",
    openai_api_key='ollama',
#    openai_api_base="http://localhost:8000/v1",
    temperature=0
)

          
#query = input("What is your question? :")
#query = "Compute the pion two-point and vector two-point functions using propagators of mass 0.01 and 0.03."
#query = "Compute the pion two-point function with DWF propagators of mass 0.01 and 0.03, and again with masses 0.02 and 0.04."

#query = """Compute the pion two-point function with DWF propagators of mass 0.01 and 0.03, and again with masses 0.02 and 0.04.
#General parameters:
#Ls=12
#M5=1.8"""

#query = """Compute the pion two-point function with DWF wall-source propagators of mass 0.01 and 0.03 and source timeslice t=0, and again with masses 0.02 and 0.04 and source timeslice t=32. Assign momentum [1,2,3,4] to the sources at timeslice t=32.
#General parameters:
#M5=1.8"""

#query = "Compute the pion two-point function"
#query = "Compute the pion two-point function with mass 0.01 for both propagators and again with mass 0.02 for both"
#query = "Compute the pion two-point and vector two-point functions using propagators of mass 0.01 and 0.03."

query = """Compute the pion two-point and vector two-point functions. In both cases use a propagator of mass 0.01 and another of mass 0.03.
Other parameters:
Use DWF quarks with M5=1.8 and Ls=12
Use the RBPrecCG solver with residual 1e-8 and default max iterations
Use the unit gauge
"""

#Only ever answer no if it did not correctly parse information you have previously provided. Do not answer no if it is simply missing information that you know and it hasn't yet asked you. You can assume that it will ask if missing any information in later stages of the workflow.

#You are a lattice QCD researcher   n agent responsible for testing an agentic workflow tool by acting in the place of the user, providing information to the workflow when queried.

#The tool you are evaluating is designed to help a user generate a job configuration for a lattice QCD measurement. This tool operates as a workflow that builds the job configuration in stages.


enableAutoEvaluate(llm, f"""
You are a lattice QCD researcher using an agentic workflow tool to help a you generate a job configuration for a lattice QCD measurement. This tool operates as a workflow that builds the job configuration in a series of stages by combining the user's query with their response to a series of questions.

You are reluctant to divulge information unless asked. You must directly answer the questions. Do not volunteer additional information unless specifically requested. Keep your output minimal and to the point, do not be overly verbose. Prefer a conversational, humanlike response format rather than a structured output.

At the end of each stage you will be asked a yes/no query as to whether the tool is correct. Answer 'y' (yes) if the information is correct and 'n' (no) otherwise. Answer yes if the information is correct even if it is incomplete according to your knowledge. Answer no only if the information contains errors.

Following a yes/no query, if you answered no you will be asked a follow-up question to explain why. Your response should describe what details of the tool's output are incorrect and what needs to be done to fix them. Only describe what is wrong, do not volunteer any information that it did not ask for. Be very concise in your response favoring brevity over clarity.


Stage-specific guidance
-----------------------

Sources
- Ensure your response clarifies which observable the source is associated with.


The initial query to the tool is as follows:
===============================
{query}
===============================

Here is the extra information you will need to completely specify the job:
- The pion two-point function should be measured using wall source propagators with one at t=0 and the other at t=32
- The vector two-point function should be measured using a point source propagator at [0,0,0,0] and a wall source propagator at t=0
- The tool should identify 3 propagators:
1. a wall source propagator at t=0 and mass 0.01
2. a wall source propagator at t=32 and mass 0.03
3. a point source propagator at [0,0,0,0] and mass 0.03

Only provide this extra information when it is explicitly requested by the tool.
""")
print(query)
agent(query, llm, reload_state=False)

#agent(query, zerot_llm, reload_state=True, ckpoint_file="state.json")


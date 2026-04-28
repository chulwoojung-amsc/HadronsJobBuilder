from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage
from langchain_core.messages import (
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

import femtomeas.meas_config_agent.common as common
from femtomeas.meas_config_agent.agent import agent
import io

#For agent-based verification
auto_eval_messages = None
auto_eval_ostrm = None  
auto_eval_model = None
auto_eval_sys = None

def autoEvalPrint(*args, **kwargs):
    global auto_eval_ostrm
    print(*args, *kwargs, file=auto_eval_ostrm)
def autoEvalInput(query):
    global auto_eval_ostrm
    global auto_eval_messages
    auto_eval_messages.append(HumanMessage(auto_eval_ostrm.getvalue()))
    auto_eval_ostrm.close()
    auto_eval_ostrm = io.StringIO()

    auto_eval_messages.append(HumanMessage(query))
    print("TO EVAL AGENT:", auto_eval_messages[-2].content,"\n",auto_eval_messages[-1].content)
        
    msg = [ SystemMessage(auto_eval_sys) ] + auto_eval_messages 
    ret = auto_eval_model.invoke(msg).content
    print(f"EVAL AGENT RESPONSE |{ret}|")
    return ret

def enableAutoEvaluate(model, sys):
    global auto_eval_model
    global auto_eval_sys
    global auto_eval_messages
    global auto_eval_ostrm
    auto_eval_model = model
    auto_eval_sys = sys
    auto_eval_ostrm = io.StringIO()
    auto_eval_messages = []
    common.print_func = autoEvalPrint
    common.input_func = autoEvalInput


warm_llm = ChatOpenAI(
    model="gpt-oss-120b-GGUF",
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

amsc_llm_0t = ChatOpenAI(
    model="gpt-oss-120b",
    base_url="https://api.i2-core.american-science-cloud.org/",
    temperature=0
)

llm = amsc_llm_0t

query = """Compute the pion two-point and vector two-point functions. In both cases use a propagator of mass 0.01 and another of mass 0.03.
Other parameters:
Use DWF quarks with M5=1.8 and Ls=12
Use the RBPrecCG solver with residual 1e-8 and default max iterations
Use the unit gauge
"""

enableAutoEvaluate(llm, f"""
You are a lattice QCD researcher using an agentic workflow tool to help a you generate a job configuration for a lattice QCD measurement. This tool operates as a workflow that builds the job configuration in a series of stages by combining the user's query with their response to a series of questions.

You are reluctant to divulge information unless asked. You must directly answer the questions. Do not volunteer additional information unless specifically requested. Keep your output minimal and to the point, do not be overly verbose. Prefer a conversational, humanlike response format rather than a structured output.

End-of-stage instructions
-------------------------
At the end of each stage you will be asked a yes/no query (indicated by a question ending in [y/n]) as to whether the tool is correct. Answer 'y' (yes) if the information is correct and 'n' (no) otherwise. Answer 'y' if the information is correct even if it is incomplete according to your knowledge. Answer 'n' only if the information contains errors. Your response to this question must be either 'y' or 'n' with no extra characters including newlines.

Following this [y/n] query, if you answered no you will be asked a follow-up question to explain why. Your response should describe what details of the tool's output are incorrect and what needs to be done to fix them. Only describe what is wrong, do not volunteer any information that it did not ask for. Be very concise in your response favoring brevity over clarity.

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
#agent(query, llm, reload_state=False)
agent(query, llm)



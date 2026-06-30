from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage
from langchain.messages import (
    SystemMessage,
    HumanMessage,
    ToolCall,
    AIMessage
)

import femtomeas.agent_common.common as common
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
    auto_eval_messages.append(AIMessage(auto_eval_ostrm.getvalue()))
    auto_eval_ostrm.close()
    auto_eval_ostrm = io.StringIO()

    auto_eval_messages.append(AIMessage(query))
    print("TO EVAL AGENT:", auto_eval_messages[-2].content,"\n",auto_eval_messages[-1].content)
        
    msg = [ SystemMessage(auto_eval_sys) ] + auto_eval_messages 
    ret = auto_eval_model.invoke(msg).content

    #gpt-oss-120b with the AmSC LLM services sometimes runs ahead of itself with intervening blocks of reasoning output directly into the content in xml-like tags.
    #However it seems that the content before the first tag is the intended output.
    if "<reasoning>" in ret:
        ret = ret[:ret.find('<reasoning>')]

    auto_eval_messages.append(HumanMessage(ret))

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
    openai_api_base="http://localhost:8000/v1"
)

zerot_llm = ChatOpenAI(
    model="gpt-oss-120b-GGUF",
    openai_api_key="sk-local",
    openai_api_base="http://localhost:8000/v1",
    temperature=0
)

amsc_llm_0t = ChatOpenAI(
    model="gpt-oss-120b",
    base_url="https://api.i2-core.american-science-cloud.org/",
    temperature=0
)
amsc_llm_warm = ChatOpenAI(
    model="gpt-oss-120b",
    base_url="https://api.i2-core.american-science-cloud.org/",
    temperature=1
)

llm = amsc_llm_warm #amsc_llm_0t

query = """Compute the pion two-point and vector two-point functions. In both cases use a propagator of mass 0.01 and another of mass 0.03.
Other parameters:
Use DWF quarks with M5=1.8 and Ls=12
Use the RBPrecCG solver with residual 1e-8 and default max iterations
Use the unit gauge
Do not use eigenvectors
Comnpute the two-point functions using both wall-smeared propagators and unsmeared propagators.
"""

enableAutoEvaluate(llm, f"""
You are playing the role of a lattice QCD researcher who is using an agentic workflow tool to help generate a job configuration for a lattice QCD measurement. The tool operates as a workflow that builds the job configuration in a series of stages based on user input.

In this role you must make *ALL* decisions on values for parameters and instance types. You MUST NOT ask the tool to make decisions for you.

The tool progresses through a series of stages: "IDENTIFY OBSERVABLES", "ACTIONS", "SOURCES", "EIGENSOLVERS", "SOLVERS", "PROPAGATORS", "SMEARED PROPAGATORS", "OBSERVABLE CONFIGURATIONS", "GAUGE CONFIGURATIONS".
Once each stage has completed, you will not be able to make changes to the objects in that stage. For example, once you get to "OBSERVABLE CONFIGURATIONS" you cannot add or modify any sources. Thus, when a stage completes, you *must* ensure that all the object instances you require are being created.

--------------------                   
Rules for responses 
--------------------
- You are reluctant to divulge information unless asked. 
- You must directly answer the questions provided by the tool. 
- Do not volunteer additional information unless specifically requested. 
- Keep your output minimal and to the point, do not be overly verbose. Prefer a conversational, humanlike response format rather than a structured output.
- If you don't know what value to use for a parameter, you can either choose a value or ask the tool to suggest a value.                    
- You can ask the tool to suggest appropriate values for parameters, but do not ask the tool to make decisions for you. You and not the tool must make all decisions.
- *Never* ask the tool what value it would "like to use" or "wants". Never ask the tool to "specify" or "give" a value. The tool is only allowed to offer suggestions. If you want a suggestion, ask the tool to "suggest" a value.
- If you specify an instance type but the tool misunderstands and selects a different type, you *must* correct the tool. Always check that the tool is asking questions about the correct type.
- If you specify a parameter value but the tool misunderstands and selects a different value, you *must* correct the tool.
- Never specify names for instances unless the tool explicitly asks you too.
- Never ask the tool to confirm values.
- Never send code or JSON to the tool; your responses must be in plain text
                                                                            
-----------------
Yes/no questions
-----------------                                                         
- If asked a question that contains '[y/n]' you *MUST* respond either the single character 'y' (for yes) or the single character 'n' (for no). Do not include any other text in your response.
                   
Stage-specific guidance
-----------------------

SOURCES
- Ensure your response clarifies which observable the source is associated with.

PROPAGATORS
- This stage is for non-sink-smeared propagators. Do not mention sink smeared propagators here.

SMEARED PROPAGATORS
- This stage is for sink-smeared propagators only.                   

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

common.log_stream = open("auto_eval.log", 'w')

print("\n#########################\nHuman:\n %s" % query, file=common.log_stream, flush=True)  

agent(query, llm)

common.log_stream.close()



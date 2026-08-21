from pydantic import BaseModel, Field
from .routing_agent_registries import getWorkflowCallables, BaseRoutingHandle
from .agent_base import promptStringList
import asteval
from langchain.messages import HumanMessage
from .common import queryYesNo, prettyPrintPydantic, prettyPrintPythonCode, getStructuredResponse, callModelWithStructuredOutput, Print as AgentPrint, Input as AgentInput, InputMulti as AgentInputMulti    
from .agent_base import parameterAgent

class AgentOutput(BaseModel):
    code: str = Field(..., description="Python code for performing the required steps")

def routingAgent(llm_model, registry_name :  str, role_header : str | None = None,
                 user_query_rules : list[str] = [], code_rules : list[str] = [],
                 additional_sys_prompt_content = "",
                 tools=[],
                 node_enactor=lambda func, args: func(*args) ):
    manifest = getWorkflowCallables(registry_name).workflowFunctionManifest()

    print("MANIFEST\n", manifest)

    if role_header is None:
        role_header = "creating a Python code snippet that plans a workflow based on instructions provided by the user during your conversation."

    code_rules = [
"Do not import any Python modules or functions; assume that the functions in the registry have already been imported.",
"Handles must all be consumed within the code snippet; do not store them in any output structures (lists, dictionaries, etc)",
"Your snippet should only perform the user's instructions and nothing else. Do not add code to write outputs.",
"The output handles from your code snippet should be stored in an array named 'results'. Only store the handles for the outputs (i.e. the last calls in any given chain), not the intermediaries.",
"If the user specifies that they do not want any steps in the workflow, output an empty array named 'results'."
    ] + code_rules

    role = f"""{role_header}

When you begin your workflow, ask the user for instructions.

Your code snippet must employ the functions in the "Registry" below. Follow the "Code rules" below. 

Do not ask the user to confirm or accept your code.
Never ask the user to provide the code.

If the user asks you questions you must answer them.

If there are multiple valid sequences to achieve the user's workflow, ask the user questions to determine which they would like to use. Do not ask the user to confirm steps that are mandatory.

-------------
Registry
-------------
{manifest}

-------------
Code rules
-------------
{promptStringList(code_rules,0)}
  
-----------
Tool usage
-----------
- If a tool for obtaining a list returns an empty list, interpret this as meaning no elements currently exist. NEVER repeatedly call the same tool over and over if it returns an empty list.

{additional_sys_prompt_content}
"""

    user_query_rules = [
    "If there are optional steps, you MUST ask the user if they want to perform those steps; NEVER make assumptions."] + user_query_rules

    def printCode(obj):
        return prettyPrintPydantic(obj.code)

    graphs = None
    
    def validator(obj):        
        reg_dict = getWorkflowCallables(registry_name).workflowFunctionSymtable()

        symtable_in = asteval.make_symbol_table(use_numpy=False, **reg_dict)
        aeval = asteval.Interpreter(symtable=symtable_in)        
        aeval(obj.code)

        errors = ""
        if len(aeval.error)>0:
            for err in aeval.error:
                e = err.get_error()
                errors = errors + f"{e[0]}:{e[1]}\n"
        if len(errors) > 0:
            print("USED INSTANCE CODE ERRORS", errors)
            return False, HumanMessage(f"Running your use_instance_code code produced error(s): {errors}")    

        if "results" not in aeval.symtable:
            print("RESULTS NOT IN CODE")
            return False, HumanMessage("Your code must produce an array of handles named 'results'")
        if not isinstance(aeval.symtable['results'], list):
            print("RESULTS NOT LIST")
            return False, HumanMessage("Your 'results' output must be an array")
        for h in aeval.symtable['results']:
            if not isinstance(h, BaseRoutingHandle):
                print("RESULTS ELEMENT NOT ROUTINGHANDLE")
                return False, HumanMessage("Your 'results' output array must contain only handles")

        nonlocal graphs
        graphs = [ h.parent_node for h in aeval.symtable['results'] ]
        return True, ""

    obj = parameterAgent(llm_model, AgentOutput, role, tools=tools, additional_user_query_rules=user_query_rules, human_validation_output_formatter=printCode, validator=validator)

    assert graphs is not None
    cache={}
    #Evaluate all output handles with caching in case they are branches from the same chain

    for g in graphs:
        g.evalWithCache(cache, enactor=node_enactor)

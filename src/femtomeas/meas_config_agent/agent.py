from femtomeas.agent_common.common import *
from femtomeas.agent_common.agent_base import parameterAgent, parameterModelCall

from typing import Tuple, TypeVar, ClassVar

from langchain_core.messages import BaseMessage
from langchain.messages import (
    SystemMessage,
    HumanMessage,
    ToolCall,
    AIMessage
)

from pydantic import BaseModel, Field, ConfigDict, NonNegativeInt, TypeAdapter, PositiveFloat, PositiveInt, create_model, model_validator
from typing import Literal, Union, List, Optional, Tuple, Any
import json
from femtomeas.agent_common.common import getUserInput, provideInformationToUser, queryYesNo, prettyPrintPydantic, getStructuredResponse, Print as AgentPrint, Input as AgentInput
from langchain.tools import tool
import re
import asteval
from femtomeas.agent_common.callgraph import Node
from femtomeas.workflow_manager.manager_config import readManagerConfigFile
from femtomeas.agent_common.agent_config import readI2APIkey
from .state import State, checkpointState
from .agent_workflows import BaseGroup, BaseGroupHandle, checkValidNewGroupName, getCurrentState, reserved_names, initializeState
from .agent_workflow_globals import registerWorkflowOperation, workflowFunctionManifest, workflowFunctionSymtable, getUniqueIdx
from femtomeas.agent_common.callgraph import Node
from .gauge import identifyGaugeConfigs

#Ensure you import all modules that define agent actions here so that they are registered
from . import action_config
from . import source_config
from . import source_config
from . import propagator_config
from . import observable_config
from . import smeared_prop_config

@registerWorkflowOperation()
def mergeGroup[T: BaseGroupHandle](group_name: str, a: T, b : T, *other_group_handles : T)->T:
    """Create a new group by merging two or more groups
    Arguments:
        group_name: The name of the merged group
        a, b, *other_group_handles: handles associated with the groups that will be merged
    Return:
        A handle to the new group        
    """
    def doit(*args):
        state, _ = getCurrentState()
        print("MERGE DOIT ", len(args))
        group_type = None
        input_groups_code = ""
        code = f"{group_name}=list(set([" #ensure unique
        first = True
        for name in args:
            assert name in state.groups
            group = state.groups[name]
            if group_type is None:
                group_type = type(group)
            else:
                assert isinstance(group, group_type)

            print("MERGE ARG TYPE", group_type.__name__, group.code)
            input_groups_code += group.code + "\n"
            code += (", " if not first else "") + f"*{name}"
            first=False
        code += "]))"
        code = input_groups_code + code
        print("MERGED CODE ",code)
        state.groups[group_name] = group_type(code=code)
        return group_name
    
    checkValidNewGroupName(group_name)
    handle_type = type(a)
    for arg in [b, *other_group_handles]:
        if type(arg) is not handle_type:
            raise Exception("Handle types must all be the same")

    print("MERGE GRAPH ", len(other_group_handles) + 2)
    return handle_type(group_name, Node(f"mergeGroup_{getUniqueIdx()}", lambda ga, gb, *ogb: doit(ga,gb,*ogb), input_deps=[a,b,*other_group_handles] ) )

@registerWorkflowOperation()
def retrieveGroupHandle[T: BaseGroupHandle](group_name : str)->T:
    """Retrieve a handle to an existing group by name"""
    state, _ = getCurrentState()
    assert group_name in state.groups
    group = state.groups[group_name]
    handle_type = group.handle_type
    print("RETRIEVE HANDLE ", group_name, " ", handle_type.__name__)
    return handle_type(group_name, Node(f"retrieveGroupHandle_{getUniqueIdx()}", lambda: group_name))


@tool
def listGroupHandles()->List[Tuple[str,str] ] | None:
    """List the existing group names and their corresponding handle types
    Return: A list of tuples containing the group name and group handle type, None if no groups exist
    """
    state, _ = getCurrentState()
    ret = [(k, type(v).__name__) for k,v in state.groups.items()]
    print("LISTGROUPHANDLES", ret)
    return ret if len(ret) > 0 else None

class AgentOutput(BaseModel):
    code: str = Field(..., description="Python code for performing the required steps")

# When executed later, these functions call subagents who will query the user for module parameters and types and then append "module" class instances to a hidden internal state. Your only responsibility is to *plan* through your code the sequence of subagent calls as required by the user. You are not responsible for obtaining the parameters of those module instances or their types. These parameters are not required to call the function.
# - When your workflow is complete, review the user's messages and compile any relevant information about module instances into the "user_info" parameter of a function, if it exists.    
#     - This field is NOT REQUIRED for a valid workflow. The subagents will ask the user for any parameters it does not know.
#     - You must choose the value of this parameter, based on user input.
#     - Only pass information relevant to the module type associated with the function. 
#     - Pass ALL information the user has provided about the parameters of those module instances and their types, even if the same information is passed multiple times.
#     - If the user does not provide any information, do not specify this parameter. NEVER guess this parameter. You MUST only specify this parameter if the user has provided relevant information.
#     - NEVER ask the user to provide user_info.


def subWorkflowAgent(llm_model, checkpoint_state: Tuple[bool, str] = (False, "")  ):    
    print("MANIFEST\n", workflowFunctionManifest())


    do_checkpoint, checkpoint_file = checkpoint_state

    role = f"""creating a Python code snippet that plans the *logical* structure of a lattice QCD measurement workflow based on instructions provided by the user during your conversation. Parameters and other details are not needed at this stage.

When you begin your workflow, ask the user for instructions.

Your code snippet must employ the functions in the "Registry" below. Follow the "Code rules" below. 

Do not ask the user to confirm or accept your code.
Never ask the user to provide the code.
NEVER ask the user to provide parameters or types used in the code, other than group names.

If the user asks you questions you must answer them.

If there are multiple valid sequences to achieve the user's workflow, ask the user questions to determine which they would like to use. Do not ask the user to confirm steps that are mandatory.

-------------
Registry
-------------
{workflowFunctionManifest()}

-------------
Code rules
-------------
- Do not import any Python modules or functions; assume that the functions in the registry have already been imported.
- Handles must all be consumed within the code snippet; do not store them in any output structures (lists, dictionaries, etc)
- Your snippet should only perform the user's instructions and nothing else. Do not add code to write outputs.
- The output handles from your code snippet should be stored in an array named 'results'. Only store the handles for the outputs (i.e. the last calls in any given chain), not the intermediaries.
- If the user specifies that they do not want any steps in the workflow, output an empty array named 'results'.
- For user_info, only include information that the user voluntarily provides without you asking. NEVER guess or hallucinate its content. NEVER ask the user about this parameter.
  e.g. If the user says "Create an action group with DWF fermions", set user_info to "User specified DWF fermions"
  
-----------
Tool usage
-----------
- If a tool for obtaining a list returns an empty list, interpret this as meaning no elements currently exist. NEVER repeatedly call the same tool over and over if it returns an empty list.

---------
Group names
---------
The following group names are reserved and cannot be used: f{reserved_names}

"""

    user_query_rules = [
    "You can *ONLY* ask the user questions about the sequence of Registry function calls and the group names, and NOTHING ELSE.",
    "You MUST NOT ask any question that does not pertain to the order of Registry function calls or their group names",         
    "NEVER ask the user to provide details on groups or their parameters.",
    "NEVER ask the user for module parameters or types.",
    "****NEVER***** ask the user about parameters of a module.",
    "NEVER guess the parameters of the modules created by a function and attempt to obtain them from the user.",
    "If there are optional steps, you MUST ask the user if they want to perform those steps; NEVER make assumptions.",
    "You are allowed to choose group names if they have not been specified by the user, unless otherwise directed. Never tell the user that you cannot choose a group name for them.",
    "Any questions about module instance types ARE FORBIDDEN."
    ]

    def printCode(obj):
        return prettyPrintPydantic(obj.code)

    graphs = None
    
    def validator(obj):        
        reg_dict = workflowFunctionSymtable()

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
            if not isinstance(h, BaseGroupHandle):
                print("RESULTS ELEMENT NOT GROUPHANDLE")
                return False, HumanMessage("Your 'results' output array must contain only handles")

        nonlocal graphs
        graphs = [ h.parent_node for h in aeval.symtable['results'] ]
        return True, ""

    tools=[listGroupHandles]
    obj = parameterAgent(llm_model, AgentOutput, role, tools=tools, additional_user_query_rules=user_query_rules, human_validation_output_formatter=printCode, validator=validator)

    assert graphs is not None
    cache={}
    #Evaluate all output handles with caching in case they are branches from the same chain

    #enactor ensures checkpointing after every node
    def enactor(f, a): 
        ret = f(*a)
        if do_checkpoint:
            state, _ = getCurrentState()
            checkpointState(state, checkpoint_file)
        return ret

    for g in graphs:
        g.evalWithCache(cache, enactor=enactor)

def measConfigAgent(llm_model, 
                    input_state : State | None =None, 
                    checkpoint_state: Tuple[bool, str] = (False, "")  )->State:
    """
    checkpoint_state: Tuple ( True/False indicating whether to checkpoint,   filename )
    """
    initializeState(llm_model, input_state)

    do_continue = True
    while do_continue:
        subWorkflowAgent(llm_model, checkpoint_state)
        do_continue = queryYesNo("Would you like to create any more workflow stages?")

    #Finally, obtain the gauge configurations on which to perform the workflow
    state, _ = getCurrentState()
    state.gauge = identifyGaugeConfigs(llm_model, [HumanMessage("Perform your workflow")])

    #Checkpoint and return
    do_checkpoint, checkpoint_file = checkpoint_state
    if do_checkpoint:
        checkpointState(state, checkpoint_file)
    return state        
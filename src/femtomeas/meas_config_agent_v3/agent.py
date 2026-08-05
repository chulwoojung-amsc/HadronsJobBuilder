import os
from femtomeas.agent_common.common import *
from femtomeas.agent_common.print_pydantic_meta import llm_json_text
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
from langchain.agents.structured_output import ToolStrategy, ProviderStrategy
from langchain.agents import create_agent
from langchain.agents.middleware import before_model, after_model, AgentState, dynamic_prompt, ModelRequest
import json
from femtomeas.agent_common.common import getUserInput, provideInformationToUser, queryYesNo, prettyPrintPydantic, getStructuredResponse, Print as AgentPrint, Input as AgentInput
from femtomeas.workflow_manager.api_general import getKnownMachines, getUserAccountProjects, getMachineQueues
from langchain.tools import tool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.runtime import Runtime
import re
import traceback

from typing import Callable, Self
from typing_extensions import NotRequired
from femtomeas.agent_common.agent_base import listEnumerateStr, indentExceptFirst, promptStringList

import asteval
from femtomeas.agent_common.callgraph import Node
import sys
from femtomeas.workflow_manager.manager_config import readManagerConfigFile
from langchain_openai import ChatOpenAI
from femtomeas.agent_common.agent_config import readI2APIkey
from .state import State, checkpointState
from .agent_workflows import BaseGroup, BaseGroupHandle, registerWorkflowOperation, checkValidNewGroupName, getUniqueIdx, addReservedName, getCurrentState, function_manifest, GroupTypes, GroupHandleTypes, reserved_names, initializeState, registry
from femtomeas.agent_common.callgraph import Node

#Ensure you import all modules that define agent actions here so that they are registered
from . import action_config
from . import source_config
from . import source_config
from . import propagator_config
from . import observable_config
from . import smeared_prop_config


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

registerWorkflowOperation(mergeGroup)

def retrieveGroupHandle[T: BaseGroupHandle](group_name : str)->T:
    """Retrieve a handle to an existing group by name"""
    state, _ = getCurrentState()
    assert group_name in state.groups
    group = state.groups[group_name]
    handle_type = group.handle_type
    print("RETRIEVE HANDLE ", group_name, " ", handle_type.__name__)
    return handle_type(group_name, Node(f"retrieveGroupHandle_{getUniqueIdx()}", lambda: group_name))

registerWorkflowOperation(retrieveGroupHandle)


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


def subWorkflowAgent(llm_model, checkpoint_state: Tuple[bool, str] = (False, "")  ):
    do_checkpoint, checkpoint_file = checkpoint_state

    role = f"""creating a code snippet that performs the instructions provided by the user during your conversation.

When you begin your workflow, ask the user for instructions.

Your code snippet must employ the functions in the "Registry" below to construct these module instances for this particular observable. These functions act upon a hidden internal state and query the user internally. Your focus should only be on calling these tools in the appropriate order. Follow the "Code rules" below. 

Do not ask the user to confirm or accept your code.
Never ask the user to provide the code.

If the user asks you questions you must answer them.

-------------
Registry
-------------
{function_manifest()}

-------------
Code rules
-------------
- Do not import any modules or functions; assume that the functions in the registry have already been imported.
- Functions in the registry act on a hidden internal state. The inputs and outputs are merely handles for chaining the logic. Handles must all be consumed within the code snippet; do not store them in any output structures (lists, dictionaries, etc)
- Your snippet should only perform the user's instructions and nothing else. Do not add code to write outputs.
- The output handles from your code snippet should be stored in an array named 'results'. Only store the handles for the outputs (i.e. the last calls in any given chain), not the intermediaries.

-----------
Tool usage
-----------
- The workflow you define may be one of many. You have been provided tools (listGroupHandles, listWorkflows, retrieveWorkflowCode) that you can use to obtain information about previously-created groups and workflows.
- If a tool for obtaining a list returns an empty list, interpret this as meaning no elements currently exist. NEVER repeatedly call the same tool over and over if it returns an empty list.

---------
Group names
---------
The following group names are reserved and cannot be used: f{reserved_names}

"""

    user_query_rules = [
    "You can only ask the user questions about the sequence of registry function calls", 
    "NEVER ask the user to provide details on groups or their parameters.",
    "If there are optional steps, you MUST ask the user if they want to perform those steps; NEVER make assumptions.",
    "You are allowed to choose group names if they have not been specified by the user, unless otherwise directed. Never tell the user that you cannot choose a group name for them."
    ]

    def printCode(obj):
        return prettyPrintPydantic(obj.code)

    graphs = None
    
    def validator(obj):
        reg_dict = { f.__name__ : f for f in registry }   

        symtable_in = asteval.make_symbol_table(use_numpy=False, **reg_dict) #**registry)
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
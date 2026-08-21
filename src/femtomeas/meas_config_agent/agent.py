from femtomeas.agent_common.common import *
from femtomeas.agent_common.routing_agent import routingAgent

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
from .agent_workflow_base import BaseRoutingHandle, BaseGroup
from .agent_workflows import checkValidNewGroupName, getCurrentState, initializeState
from .agent_workflow_globals import registerWorkflowOperation, getUniqueIdx
from .gauge import identifyGaugeConfigs

#Ensure you import all modules that define agent actions here so that they are registered
from . import action_config
from . import source_config
from . import source_config
from . import propagator_config
from . import observable_config
from . import smeared_prop_config

@registerWorkflowOperation()
def mergeGroup[T: BaseRoutingHandle](group_name: str, a: T, b : T, *other_group_handles : T)->T:
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
    return handle_type(Node(f"mergeGroup_{getUniqueIdx()}", lambda ga, gb, *ogb: doit(ga,gb,*ogb), input_deps=[a,b,*other_group_handles] ) )

@registerWorkflowOperation()
def retrieveGroupHandle[T: BaseRoutingHandle](group_name : str)->T:
    """Retrieve a handle to an existing group by name"""
    state, _ = getCurrentState()
    assert group_name in state.groups
    group = state.groups[group_name]
    handle_type = group.handle_type
    print("RETRIEVE HANDLE ", group_name, " ", handle_type.__name__)
    return handle_type(Node(f"retrieveGroupHandle_{getUniqueIdx()}", lambda: group_name))


@tool
def listGroupHandles()->List[Tuple[str,str] ] | None:
    """List the existing group names and their corresponding handle types
    Return: A list of tuples containing the group name and group handle type, None if no groups exist
    """
    state, _ = getCurrentState()
    ret = [(k, type(v).__name__) for k,v in state.groups.items()]
    print("LISTGROUPHANDLES", ret)
    return ret if len(ret) > 0 else None




def subWorkflowAgent(llm_model, checkpoint_state: Tuple[bool, str] = (False, "")  ):    
    do_checkpoint, checkpoint_file = checkpoint_state


    def enactor(f, a): 
        ret = f(*a)
        if do_checkpoint:
            state, _ = getCurrentState()
            checkpointState(state, checkpoint_file)
        return ret

    additional_sys_prompt_content = """
---------
Group names
---------
The following group names are reserved and cannot be used: f{reserved_names}
"""

    code_rules = [
"""For user_info, only include information that the user voluntarily provides without you asking. NEVER guess or hallucinate its content. NEVER ask the user about this parameter.
   e.g. If the user says "Create an action group with DWF fermions", set user_info to "User specified DWF fermions"""
    ]

    user_query_rules = [
        "NEVER ask the user to provide parameters or types used in the code, other than group names.",
        "You can *ONLY* ask the user questions about the sequence of Registry function calls and the group names, and NOTHING ELSE.",
        "You MUST NOT ask any question that does not pertain to the order of Registry function calls or their group names",         
        "NEVER ask the user to provide details on groups or their parameters.",
        "NEVER ask the user for module parameters or types.",
        "****NEVER***** ask the user about parameters of a module.",
        "NEVER guess the parameters of the modules created by a function and attempt to obtain them from the user.",
        
        "You are allowed to choose group names if they have not been specified by the user, unless otherwise directed. Never tell the user that you cannot choose a group name for them.",
        "Any questions about module instance types ARE FORBIDDEN."
        ]

    role_header = "creating a Python code snippet that plans the *logical* structure of a lattice QCD measurement workflow based on instructions provided by the user during your conversation. Parameters and other details are not needed at this stage."

    tools=[listGroupHandles]

    routingAgent(llm_model, "meas_config_agent", role_header=role_header, user_query_rules=user_query_rules, code_rules=code_rules, additional_sys_prompt_content=additional_sys_prompt_content, tools=tools, node_enactor=enactor)


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
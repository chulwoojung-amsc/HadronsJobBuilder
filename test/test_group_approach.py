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
from inspect import signature, getdoc, currentframe
import asteval
from femtomeas.agent_common.callgraph import Node
import sys
from femtomeas.workflow_manager.manager_config import readManagerConfigFile
from langchain_openai import ChatOpenAI
from femtomeas.agent_common.agent_config import readI2APIkey

counter = 0 #for unique indexing of graph nodes
def getUniqueIdx():
    global counter
    counter += 1
    return counter - 1

def getTaskInfo(frame):
    return f"{frame.f_code.co_name} : {getdoc(globals()[frame.f_code.co_name])}"

def funcWrapInfo(func, info_str):
    """Bind an info string from the router agent to the function for later evaluation"""
    return lambda llm_model, observable_tag, _, state: func(llm_model, observable_tag, info_str, state) 


class State:
    def __init__(self):
        self.groups = {}
        self.workflows = {}

state = State()


def getGroupInstance(group_name: str, group_type: type):
    assert group_name in state.groups
    r = state.groups[group_name]
    assert isinstance(r, group_type)
    return r

class BaseGroupHandle:
    def __init__(self, group_name : str, parent_node : Node):
        self.group_name = group_name
        self.parent_node = parent_node

class ActionModel(BaseModel):
    action_type: str = Field(..., description="The type of the action")
    mass: float = Field(..., description="The fermion mass")


class ActionGroupHandle(BaseGroupHandle):
    pass

class ActionGroup(BaseModel):
    handle_type : ClassVar[type] = ActionGroupHandle
    code: str = Field(..., description="Code for generating the list of action instances in the group")


def createActionGroup(group_name: str)->ActionGroupHandle:
    def doit(group_name: str):
        #agent builds a group, adding new action instances as needed
        assert group_name not in state.groups
        state.groups[group_name] = ActionGroup(code = f"{group_name} = ['action1', 'action2']")    
        return group_name

    assert group_name not in state.groups
    return ActionGroupHandle(group_name, Node(f"createActionGroup_{getUniqueIdx()}", lambda: doit(group_name) ) )

class SolverGroupHandle(BaseGroupHandle):
    pass

class SolverGroup(BaseModel):
    handle_type : ClassVar[type] = SolverGroupHandle
    code: str = Field(..., description="Code for generating the list of solver instances in the group")


    
def createSolverGroup(group_name: str, actions: ActionGroupHandle)->SolverGroupHandle:
    assert group_name not in state.groups
    def doit(group_name, gactions_group_name: str):                        
        state.groups[group_name] = SolverGroup(code = f"{group_name} = ['solver1', 'solver2']")    
        return group_name
   
    return SolverGroupHandle(group_name, Node(f"createSolverGroup_{getUniqueIdx()}", lambda gactions: doit(group_name, gactions), input_deps=[actions] ) )

class SourceGroupHandle(BaseGroupHandle):
    pass

class SourceGroup(BaseModel):
    handle_type : ClassVar[type] = SourceGroupHandle
    code: str = Field(..., description="Code for generating the list of source instances in the group")


def createSourceGroup(group_name: str)->SourceGroupHandle:
    def doit(group_name: str):
        #agent builds a group, adding new source instances as needed
        assert group_name not in state.groups
        state.groups[group_name] = SourceGroup(code = f"{group_name} = ['source1', 'source2']")    
        return group_name

    assert group_name not in state.groups
    return SourceGroupHandle(group_name, Node(f"createSourceGroup_{getUniqueIdx()}", lambda: doit(group_name) ) )

class PropagatorGroupHandle(BaseGroupHandle):
    pass

class PropagatorGroup(BaseModel):
    handle_type : ClassVar[type] = PropagatorGroupHandle
    code: str = Field(..., description="Code for generating the list of propagator instances in the group")

    
def createPropagatorGroup(group_name: str, sources: SourceGroupHandle, solvers: SolverGroupHandle)->PropagatorGroupHandle:
    assert group_name not in state.groups
    def doit(group_name, gsources_group_name: str, gsolvers_group_name: str):                        
        state.groups[group_name] = PropagatorGroup(code = f"{group_name} = ['prop1', 'prop2']")    
        return group_name
   
    return PropagatorGroupHandle(group_name, Node(f"createPropagatorGroup_{getUniqueIdx()}", lambda gsources, gsolvers: doit(group_name, gsources, gsolvers), input_deps=[sources,solvers] ) )


class Meson2ptGroupHandle(BaseGroupHandle):
    pass

class Meson2ptGroup(BaseModel):
    handle_type : ClassVar[type] = Meson2ptGroupHandle
    code: str = Field(..., description="Code for generating the list of meson 2pt function instances in the group")

def createMeson2ptGroup(group_name: str, propagators: PropagatorGroupHandle)->Meson2ptGroupHandle:
    """Create a Meson2ptGroup and return its handle

The meson two-point function (aka meson correlator) is used to describe the lattice propagation of a meson such as a pion or kaon        
    """
    assert group_name not in state.groups
    def doit(group_name, gprops_group_name: str): 
        state.groups[group_name] = Meson2ptGroup(code = f"{group_name} = ['meson2pt1']")
        return group_name
   
    return Meson2ptGroupHandle(group_name, Node(f"createMeson2ptGroup_{getUniqueIdx()}", lambda gprops: doit(group_name, gprops), input_deps=[propagators] ) )





GroupHandleTypes = TypeVar("GroupHandleTypes", bound=BaseGroupHandle)

def mergeGroup[T: BaseGroupHandle](group_name: str, a: T, b : T, *other_group_handles : T)->T:
    """Create a new group by merging two or more groups
    Arguments:
        group_name: The name of the merged group
        a, b, *other_group_handles: handles associated with the groups that will be merged
    Return:
        A handle to the new group        
    """
    def doit(*args):
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

    assert isinstance(group_name, str)
    handle_type = type(a)
    for arg in [b, *other_group_handles]:
        if type(arg) is not handle_type:
            raise Exception("Handle types must all be the same")

    print("MERGE GRAPH ", len(other_group_handles) + 2)
    return handle_type(group_name, Node(f"mergeGroup_{getUniqueIdx()}", lambda ga, gb, *ogb: doit(ga,gb,*ogb), input_deps=[a,b,*other_group_handles] ) )

def retrieveGroupHandle[T: BaseGroupHandle](group_name : str)->T:
    """Retrieve a handle to an existing group by name"""
    assert group_name in state.groups
    group = state.groups[group_name]
    handle_type = group.handle_type
    print("RETRIEVE HANDLE ", group_name, " ", handle_type.__name__)
    return handle_type(group_name, Node(f"retrieveGroupHandle_{getUniqueIdx()}", lambda: group_name))


#registry = {  "createActionGroup" : createActionGroup, "createSolverGroup" : createSolverGroup, "createSourceGroup" : createSourceGroup, "createPropagatorGroup" : createPropagatorGroup, "mergeGroup" : mergeGroup, "retrieveGroupHandle" : retrieveGroupHandle, "createMeson2ptGroup": createMeson2ptGroup  }

registry = [  createActionGroup, createSolverGroup, createSourceGroup, createPropagatorGroup, mergeGroup, retrieveGroupHandle, createMeson2ptGroup  ]
group_types = Literal["ActionGroup", "SourceGroup", "SolverGroup", "PropagatorGroup", "Meson2ptGroup"]

@tool
def listGroupHandles(group_type: group_types)->List[str] | None:
    """List the existing group names for the given group type
    Return: A list of strings containing any group names with the provided type or None if no groups exist
    """
    print("CALLING LISTGROUPHANDLES ", group_type)
    print("EXISTING ", [(k,type(v).__name__) for k,v in state.groups.items() ])

    ret = [k for k,v in state.groups.items() if type(v).__name__ == group_type ] #   isinstance(v, handle_type)
    print(ret)
    return ret if len(ret) > 0 else None

@tool
def listWorkflows()->List[str] | None:    
    """Obtain the list of existing workflows by name. Returns None if no workflows currently exist."""
    print("LISTWORKFLOWS", list(state.workflows.keys()))
    return list(state.workflows.keys()) if len(state.workflows) > 0 else None

@tool
def retrieveWorkflowCode(workflow_name: str)->str:    
    """Retrieve the code snippet associated with an existing workflow
    workflow_name: Must be a valid workflow name obtained from listWorkflows or otherwise
    """
    print("RETRIEVEWORKFLOWCODE ",workflow_name)
    assert workflow_name in state.workflows
    return state.workflows[workflow_name][1]

# def function_manifest():
#     lines = []
#     for name, fn in registry.items():
#         sig = signature(fn)
#         doc = getdoc(fn) or ""
#         lines.append(f"- {name}{sig}: {doc}")
#     return "\n".join(lines)

def function_manifest():
    lines = []
    for fn in registry:
        name = fn.__name__
        sig = signature(fn)
        doc = getdoc(fn) or ""
        lines.append(f"- {name}{sig}: {doc}")
    return "\n".join(lines)

class AgentOutput(BaseModel):
    code: str = Field(..., description="Python code for performing the required steps")


def subWorkflowAgent(llm_model):
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
        symtable_in = asteval.make_symbol_table(use_numpy=False, **registry)
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

    obj = parameterAgent(llm_model, AgentOutput, role, tools=[listGroupHandles,listWorkflows, retrieveWorkflowCode], additional_user_query_rules=user_query_rules, human_validation_output_formatter=printCode, validator=validator)

    assert graphs is not None
    cache={}
    #Evaluate all output handles with caching in case they are branches from the same chain
    for g in graphs:
        g.evalWithCache(cache)

    workflow_name = AgentInput("Provide a name for this workflow")
    state.workflows[workflow_name] = (graphs, obj.code)


if __name__ == "__main__":
    if len(sys.argv) == 1:
        print("Need param file")

    config = readManagerConfigFile(sys.argv[1])

    amsc_llm_0t = ChatOpenAI(
        model="gpt-oss-120b",
        base_url="https://api.i2-core.american-science-cloud.org/",
        temperature=0,
        api_key = readI2APIkey(config.agent.i2api_key_path)
    )

    print(function_manifest())

    # while True:
    #     subWorkflowAgent(amsc_llm_0t)

    # def testEnactor(func, args):
    #     print("ENACTING ",func.__name__, args)
    #     return func(*args)

    # ag1 = createActionGroup("ag1")
    # graph = ag1.parent_node
    # graph.eval(enactor=testEnactor)

    # ag1_h = retrieveGroupHandle("ag1")
    # graph = ag1_h.parent_node
    # graph.eval(enactor=testEnactor)
    # ag2 = createActionGroup("ag2")
    # ag3 = createActionGroup("ag3")
    # mg1 = mergeGroup("mg1",ag1,ag2,ag3)

    # graph = mg1.parent_node
    # def testEnactor(func, args):
    #     print("ENACTING ",func.__name__, args)
    #     return func(*args)

    # graph.eval(enactor=testEnactor)



    # sh = createSolverGroup("solver_group", ah)

    # graph = sh.parent_node
    # def testEnactor(func, args):
    #     print(func.__name__, args)
    #     return func(*args)

    # graph.eval(enactor=testEnactor)



    # ah = createActionGroup("action_group")
    # sh = createSolverGroup("solver_group", ah)

    # graph = sh.parent_node
    # def testEnactor(func, args):
    #     print(func.__name__, args)
    #     return func(*args)

    # graph.eval(enactor=testEnactor)


# def measConfigAgent(query, llm_model, ckpoint_file="state.json", reload_state=False)-> State :
#     if reload_state and os.path.exists(ckpoint_file):
#         state = reloadStateCheckpoint(ckpoint_file)
#         query = state.query
#         print("Reloaded query from state file:", query)
#     else:
#         state = State()

#     state.query = query    
#     checkpointState(state,ckpoint_file)

#     print(state)

#     AgentPrint("Identifying observables...")
#     obs = identifyObservables(llm_model, state.query, state)

#     for o in obs.observables:
#         AgentPrint("Constructing workflow for observable ", o.obs_tag)
#         observableWorkflowAgent(o, state.query, llm_model, state)

from pydantic import BaseModel, Field
from typing import Tuple, TypeVar, ClassVar, Callable
from inspect import signature, getdoc, currentframe
from femtomeas.agent_common.callgraph import Node
from .state import State

counter = 0 #for unique indexing of graph nodes
def getUniqueIdx():
    global counter
    counter += 1
    return counter - 1

class BaseGroupHandle:
    def __init__(self, group_name : str, parent_node : Node):
        self.group_name = group_name
        self.parent_node = parent_node

class BaseGroup(BaseModel):
    pass

GroupHandleTypes = TypeVar("GroupHandleTypes", bound=BaseGroupHandle)
GroupTypes = TypeVar("GroupTypes", bound=BaseGroup)

registry = []

def registerWorkflowOperation(op: Callable):
    if op not in registry:
        registry.append(op)

def function_manifest():
    lines = []
    for fn in registry:
        name = fn.__name__
        sig = signature(fn)
        doc = getdoc(fn) or ""
        lines.append(f"- {name}{sig}: {doc}")
    return "\n".join(lines)


reserved_names = []

def addReservedName(nm: str):
    if nm not in reserved_names:
        reserved_names.append(nm)

state = State()
llm_model_glob = None

def initializeState(llm_model, input_state = None):
    global llm_model_glob, state
    llm_model_glob = llm_model
    state = input_state if input_state is not None else State()

def getCurrentState():
    return state, llm_model_glob    

def getGroupInstance(group_name: str, group_type: type):
    assert group_name in state.groups
    r = state.groups[group_name]
    assert isinstance(r, group_type)
    return r

def checkValidNewGroupName(group_name: str):
    assert group_name not in state.groups
    assert group_name not in reserved_names
    assert isinstance(group_name, str)
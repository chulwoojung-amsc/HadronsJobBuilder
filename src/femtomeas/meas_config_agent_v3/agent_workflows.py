from pydantic import BaseModel, Field
from typing import Tuple, TypeVar, ClassVar, Callable
from .agent_workflow_base import BaseGroup, BaseGroupHandle, GroupTypes, GroupHandleTypes
from .state import State
from .agent_workflow_globals import reserved_names

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
import os
from femtomeas.agent_common.common import *
from .state import *
from femtomeas.meas_config_agent.hadrons_xml import HadronsXML
from femtomeas.agent_common.print_pydantic_meta import llm_json_text
import femtomeas.workflow_manager as wfman
from .action_config import identifyActions
from .observable_info_models import observableSkills
from .observable_info import identifyObservables
from .source_config import identifySources
from .eigenvectors import setupEigenSolvers
from .solver_config import identifySolvers
from .propagator_config import identifyPropagators
from .smeared_prop_config import identifySmearedPropagators
from .observable_config import configureObservables
from .observable_config_models import ObservableConfig
from femtomeas.meas_config_agent.gauge import identifyGaugeConfigs
from femtomeas.agent_common.agent_base import parameterAgent, parameterModelCall

from typing import Tuple

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

counter = 0 #for unique indexing of graph nodes
def getUniqueIdx():
    global counter
    counter += 1
    return counter - 1

def getTaskInfo(frame):
    return f"{frame.f_code.co_name} : {getdoc(globals()[frame.f_code.co_name])}"

class ActionsBlob:
    def __init__(self, parent_node: Node):
        self.parent_node = parent_node

def TaskIdentifyActions(*,observable_tag: str)->ActionsBlob:
    """Obtain the set of actions required for a specific observable
    inputs:
        - observable_tag: An observable_tag for a specific observable
    outputs:
        - An object describing the actions required for that observable
    """    
    return ActionsBlob(Node(f"TaskIdentifyActions_{getUniqueIdx()}", identifyActions, node_info=getTaskInfo(currentframe()) ) )

class SourcesBlob:
    def __init__(self, parent_node: Node):
        self.parent_node = parent_node

def TaskIdentifySources(*,observable_tag: str)->SourcesBlob:
    """Obtain the set of sources required for a specific observable
    inputs:
        - observable_tag: An observable_tag for a specific observable
    outputs:
        - An object describing the sources required for that observable
    """        
    return SourcesBlob(Node(f"TaskIdentifySources_{getUniqueIdx()}", identifySources) )

class EigenSolversBlob:
    def __init__(self, parent_node: Node):
        self.parent_node = parent_node

def TaskIdentifyEigenSolvers(*,observable_tag: str, actions: ActionsBlob)->EigenSolversBlob:
    """A task to obtain the set of eigensolvers required for a specific observable
    inputs:
        - An observable_tag for a specific observable
        - The actions required for that observable
    outputs:
        - The eigensolvers required for that observable
    """    
    return EigenSolversBlob(Node(f"TaskIdentifyEigenSolvers_{getUniqueIdx()}", setupEigenSolvers, [actions], node_info=getTaskInfo(currentframe()) ) )

class SolversBlob:
    def __init__(self, parent_node: Node):
        self.parent_node = parent_node

def TaskIdentifySolvers(*,observable_tag: str, actions: ActionsBlob, eigensolvers: EigenSolversBlob | None)-> SolversBlob:
    """A task to obtain the set of solvers required for a specific observable
    inputs:
      - An observable_tag for a specific observable 
      - The actions required for that observable
      - (Optional) The eigenvectors required for that observable
    outputs:
      - The solvers required for that observable    
    """
    return SolversBlob(Node(f"TaskIdentifySolvers_{getUniqueIdx()}", identifySolvers, [actions, eigensolvers], node_info=getTaskInfo(currentframe()) ) )

class PropagatorsBlob:
    def __init__(self, parent_node: Node):
        self.parent_node = parent_node

def TaskIdentifyPropagators(*,observable_tag: str, sources: SourcesBlob, solvers: SolversBlob)-> PropagatorsBlob:
    """Obtain the set of propagators required for a specific observable
    inputs:
        - An observable_tag for a specific observable
        - The solvers required to compute the propagators for this observable
        - The sources required to compute the propagators for this observable
    outputs:
        - The propagators required to compute that observable
        """
    return PropagatorsBlob(Node(f"TaskIdentifyPropagators_{getUniqueIdx()}", identifyPropagators, [sources,solvers], node_info=getTaskInfo(currentframe()) ))

class SmearedPropagatorsBlob:
    def __init__(self, parent_node: Node):
        self.parent_node = parent_node

def TaskIdentifySmearedPropagators(*,observable_tag: str, propagators: PropagatorsBlob)-> SmearedPropagatorsBlob:
    """Obtain the set of smeared propagators required for a specific observable
    inputs:
        - An observable_tag for a specific observable
        - The propagators required to compute that observable        
    outputs:
        - The smeared propagators required to compute that observable (if any)
    """
    return SmearedPropagatorsBlob(Node(f"TaskIdentifySmearedPropagators_{getUniqueIdx()}", identifySmearedPropagators, [propagators], node_info=getTaskInfo(currentframe()) ))

class ObservableComputeBlob:
    def __init__(self, parent_node: Node):
        self.parent_node = parent_node

def TaskComputeObservable(*,observable_tag: str, propagators: PropagatorsBlob | None, smeared_propagators: SmearedPropagatorsBlob | None)->ObservableComputeBlob:
    """Obtain the parameters and details for computing the observable
    Supports:
        - Meson two-point functions
        - Writing propagators to disk

    inputs:
        - An observable_tag for a specific observable
        - The propagators required to compute that observable
        - The smeared propagators required to compute that observable        

    Note: this function cannot be called with both "propagators" and "smeared_propagators" as None
        
    outputs:
        - The module instances required to calculate this observable
    """
    return ObservableComputeBlob(Node(f"TaskComputeObservable_{getUniqueIdx()}", configureObservables, [propagators, smeared_propagators], node_info=getTaskInfo(currentframe()) ))


registry = {  "TaskIdentifyActions" : TaskIdentifyActions, "TaskIdentifySources" : TaskIdentifySources, "TaskIdentifyEigenSolvers" : TaskIdentifyEigenSolvers, "TaskIdentifySolvers" : TaskIdentifySolvers,
                "TaskIdentifyPropagators" : TaskIdentifyPropagators, "TaskIdentifySmearedPropagators": TaskIdentifySmearedPropagators, "TaskComputeObservable": TaskComputeObservable }

def function_manifest():
    lines = []
    for name, fn in registry.items():
        sig = signature(fn)
        doc = getdoc(fn) or ""
        lines.append(f"- {name}{sig}: {doc}")
    return "\n".join(lines)

def enactGraph(graph, observable_tag, llm_model, state):    
    graph.eval(lambda f, _: f(llm_model, observable_tag, "", state))

class AgentOutput(BaseModel):
    code: str = Field(..., description="Python code for performing the measurement workflow")


def observableWorkflowAgent(obs_instance : ObservableConfig, query, llm_model, state):
    role = f"""writing a Python code snippet for performing the measurement of the following lattice QCD observable:
{obs_instance.model_dump_json()}                        
                            
Measurement jobs are composed of module instances described alongside their parameters in an XML document that is passed to the LQCD software.

Your code snippet must employ the functions in the "Registry" below to construct these module instances for this particular observable. These functions act upon a hidden internal state and query the user internally. Your focus should only be on calling these tools in the appropriate order. Follow the "Code rules" below. 

For the "observable_tag" input of the Registry functions, use "{obs_instance.obs_tag}"

Do not ask the user to confirm or accept your code.

Module instances fall into classes: action modules, solver modules (for inverting the Dirac operator), eigensolver modules, source modules (for propagator sources), propagator modules and sink-smeared propagator modules.

Module dependencies form a directed graph. Most common observables such as two-point functions are built from modules belonging to the above classes with the following dependencies:
action -> solver
action -> eigensolver
source, solver, (optional eigensolver) -> propagator
propagator -> sink-smeared propagator
propagators / sink-smeared propagators -> observables
These dependency chains are also encapsulated in the inputs and outputs of the registry functions.

For deciding on the appropriate chain of functions, refer to the observable skill in the "Skill" section below.        

-------------
Registry
-------------
{function_manifest()}

-------------
Code rules
-------------
- Do not import any modules or functions; assume that the functions in the registry have already been imported.
- Functions in the registry act on a hidden internal state. The inputs and outputs are merely handles for chaining the logic. Handles must all be consumed within the code snippet; do not store them in any output structures (lists, dictionaries, etc)
- Your snippet should only perform the workflow and nothing else. Do not add code to write outputs.
- Some functions allowing passing in the output of a previous call to this function. Use this to update set of modules in the class for subsequent calls to the function.
- The last function call in the snippet should be to TaskComputeObservable, and the output ObservableComputeBlob object must be named "result"
-------------
Skill
-------------
{obs_instance.obs_type.skill()}
"""

    user_query_rules = [
    "You can only ask the user questions about the sequence of registry function calls. The actual module instances within each class are determined by those registry functions.", 
    "NEVER ask the user to provide details on modules or their parameters.",
    "If there are optional steps, you MUST ask the user if they want to perform those steps; NEVER make assumptions."
    ]

    def printCode(obj):
        return prettyPrintPydantic(obj.code)

    graph = None

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

        if "result" not in aeval.symtable:
            print("RESULT NOT IN CODE")
            return False, HumanMessage("Your code must produce an ObservableComputeBlob named 'result'")
        if not isinstance(aeval.symtable['result'], ObservableComputeBlob):
            print("RESULT NOT ObservableComputeBlob")
            return False, HumanMessage("'result' must be an ObservableComputeBlob instance")

        nonlocal graph
        graph = aeval.symtable['result'].parent_node #store the validated graph so we don't need to reevaluate if it is accepted
        return True, ""

    _ = parameterAgent(llm_model, AgentOutput, role, tools=[], additional_user_query_rules=user_query_rules, human_validation_output_formatter=printCode, validator=validator)

    assert graph is not None
    enactGraph(graph, obs_instance.obs_tag, llm_model, state)







def measConfigAgent(query, llm_model, ckpoint_file="state.json", reload_state=False)-> State :
    if reload_state and os.path.exists(ckpoint_file):
        state = reloadStateCheckpoint(ckpoint_file)
        query = state.query
        print("Reloaded query from state file:", query)
    else:
        state = State()

    state.query = query    
    checkpointState(state,ckpoint_file)

    print(state)

    AgentPrint("Identifying observables...")
    obs = identifyObservables(llm_model, state.query, state)

    for o in obs.observables:
        AgentPrint("Constructing workflow for observable ", o.obs_tag)
        observableWorkflowAgent(o, query, llm_model, state)

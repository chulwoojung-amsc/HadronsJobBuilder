from typing import Callable, Tuple, Dict, Any, Union
from inspect import signature, getdoc, currentframe
import inspect
from pydantic import BaseModel
from .agent_workflow_base import BaseGroupHandle
from typing import get_args, get_origin
import types
import typing

counter = 0 #for unique indexing of graph nodes
def getUniqueIdx():
    global counter
    counter += 1
    return counter - 1

#Workflow operations registry
registry = {}

def addWorkflowOperationToRegistry(op_internal: Callable, op_call: Callable):
    print("addWorkflowOperationToRegistry ", op_internal.__name__)
    if op_internal.__name__ not in registry.keys():
        registry[op_internal.__name__] = (op_internal, op_call) #first is used only for the signature

#Names that cannot be used by the agents
reserved_names = []

def addReservedName(nm: str):
    print("addReservedName ", nm)
    if nm not in reserved_names:
        reserved_names.append(nm)

#Map of instance class string (e.g. actions, sources) to type (ActionConfig, SourceConfig)
instance_class_registry = {}
def registerInstanceClass(instance_class: str, instance_model: type):
    print("registerInstanceClass ", instance_class, instance_model.__name__)
    instance_class_registry[instance_class] = instance_model

#Decorator to perform all registrations
def registerWorkflowOperation(*, instance_info: Tuple[str,type] | None = None  ):
    """
    Workflow operations are nodes of the workflow that the routing agent can plan. 
    
    For workflow operations that populate State.instances, you must provide a tuple instance_info containing the "instance_class", which is the name of array within the instance code, and the "instance_model", the Pydantic model used to parse/validate that code.
    """

    def wrap(workflow_operation: Callable):    
        sig = signature(workflow_operation)
        pcheck = {}
        
        for name, param in sig.parameters.items():            
            if param.annotation is not inspect.Parameter.empty:
                if get_origin(param.annotation) in (types.UnionType, typing.Union):
                    ptypes = get_args(param.annotation)
                    #Allow None or BaseGroupHandle derivative
                    is_handle=True
                    for t in ptypes:
                        if t is not types.NoneType and not issubclass(t, BaseGroupHandle):
                            is_handle = False
                            break
                    if is_handle:
                        pcheck[name] = ptypes                        
                elif inspect.isclass(param.annotation) and issubclass(param.annotation, BaseGroupHandle):
                    pcheck[name] = (param.annotation,)
                                

        if instance_info is not None:
            instance_class, instance_model = instance_info
            addReservedName(instance_class) #stop the agent using the instance class name for a group or instance            
            registerInstanceClass(instance_class, instance_model) #record the mapping between the class and its Pydantic model        

        def wrapOperation(*args, **kwargs):            
            sig = signature(workflow_operation)
            sargs = sig.bind(*args, **kwargs)

            for name, value in sargs.arguments.items():
                if name in pcheck:                    
                    if type(value) not in pcheck[name]:
                        raise Exception(f"Argument {name} accepts only types {pcheck[name]}")
            return workflow_operation(*args, **kwargs)                

        addWorkflowOperationToRegistry(workflow_operation, wrapOperation) #register the workflow operation so the routing agent can use it

        return wrapOperation        
    return wrap


def workflowFunctionManifest():
    lines = []
    for name, r in registry.items():
        fn = r[0] #for signature        
        sig = signature(fn)
        doc = getdoc(fn) or ""
        lines.append(f"- {name}{sig}: {doc}")
    return "\n".join(lines)

def workflowFunctionSymtable():
    return { fname : r[1] for fname, r in registry.items() } 


#Registry for models allowed for Union types within instance models
instance_model_registry: Dict[str, type[BaseModel] ] = {}

def registerInstanceModel(name: str):
    def decorator(cls: type[BaseModel]) -> type[BaseModel]:
        instance_model_registry.setdefault(name, []).append(cls)
        return cls

    return decorator

def getInstanceModel(name: str) -> Any:
    models = instance_model_registry[name]
    if not models:
        raise ValueError(f"No instance models registered for {name!r}")
    return Union[tuple(models)]
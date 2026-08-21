from .callgraph import Node
from typing import Callable
from inspect import signature, getdoc
import inspect
from typing import get_args, get_origin
import types
import typing

class BaseRoutingHandle:
    """Base class of all routing handles (the outputs of workflow stages)"""
    def __init__(self, parent_node : Node):        
        self.parent_node = parent_node

class workflowCallables:
    def __init__(self, registry_name):
        self.counter = 0
        self.registry = {} #registry of workflow operations
        self.registry_name = registry_name

    def getUniqueIdx(self):
        ret = self.counter
        self.counter += 1
        return ret

    def addWorkflowOperationToRegistry(self, op_internal: Callable, op_call: Callable):
        """Add a workflow operation to the registry
        op_call : the callable that is actually called to perform the action
        op_internal: the underlying function, used for name and signature only (for basic usage, op_call == op_internal), but separating the two allows for decorators that wrap the internal function    
        """
    
        print("addWorkflowOperationToRegistry ", self.registry_name, " " , op_internal.__name__)
        if op_internal.__name__ not in self.registry.keys():
            self.registry[op_internal.__name__] = (op_internal, op_call) #first is used only for the signature


    def workflowFunctionManifest(self):
        lines = []
        for name, r in self.registry.items():
            fn = r[0] #for signature        
            sig = signature(fn)
            doc = getdoc(fn) or ""
            lines.append(f"- {name}{sig}: {doc}")
        return "\n".join(lines)

    def workflowFunctionSymtable(self):
        return { fname : r[1] for fname, r in self.registry.items() } 
   
workflow_registries: dict[str, workflowCallables ] = {} #allow for multiple disjoint types of workflow that the agent can handle

def getWorkflowCallables(registry_name):
    if registry_name not in workflow_registries.keys():
        workflow_registries[registry_name] = workflowCallables(registry_name)
    return workflow_registries[registry_name] 


def enwrapWorkflowOperation(registry_name: str, workflow_operation: Callable):
    """Underlying functionality of a decorator that registers the workflow operation at start time and does type checking at execution time"""
    sig = signature(workflow_operation)
    pcheck = {}
    
    for name, param in sig.parameters.items():            
        if param.annotation is not inspect.Parameter.empty:
            if get_origin(param.annotation) in (types.UnionType, typing.Union):
                ptypes = get_args(param.annotation)
                #Allow None or BaseRoutingHandle derivative
                is_handle=True
                for t in ptypes:
                    if t is not types.NoneType and not issubclass(t, BaseRoutingHandle):
                        is_handle = False
                        break
                if is_handle:
                    pcheck[name] = ptypes                        
            elif inspect.isclass(param.annotation) and issubclass(param.annotation, BaseRoutingHandle):
                pcheck[name] = (param.annotation,)

    #Enwrap the workflow operation to perform a type check of input types derived from BaseRoutingHandle when the node graph is created
    def wrapper(*args, **kwargs):            
        sig = signature(workflow_operation)
        sargs = sig.bind(*args, **kwargs)

        for name, value in sargs.arguments.items():
            if name in pcheck:                    
                if type(value) not in pcheck[name]:
                    raise Exception(f"Argument {name} accepts only types {pcheck[name]}")
        return workflow_operation(*args, **kwargs)                

    getWorkflowCallables(registry_name).addWorkflowOperationToRegistry(workflow_operation, wrapper) #register the workflow operation so the routing agent can use it

    return wrapper        


#Decorator to perform all registrations
def registerWorkflowOperation(registry_name: str):
    """
    Workflow operations are nodes of the workflow that the routing agent can plan. 
    """
    def wrap(workflow_operation: Callable):
        return enwrapWorkflowOperation(registry_name, workflow_operation)
    return wrap

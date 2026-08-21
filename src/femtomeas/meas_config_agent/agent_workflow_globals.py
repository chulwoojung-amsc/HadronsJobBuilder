from typing import Callable, Tuple, Dict, Any, Union
from inspect import signature, getdoc, currentframe
from pydantic import BaseModel
from femtomeas.agent_common.routing_agent_registries import enwrapWorkflowOperation, getWorkflowCallables

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



#Decorator to perform all registrations
def registerWorkflowOperation(*, instance_info: Tuple[str,type] | None = None  ):
    """
    Workflow operations are nodes of the workflow that the routing agent can plan. 
    
    For workflow operations that populate State.instances, you must provide a tuple instance_info containing the "instance_class", which is the name of array within the instance code, and the "instance_model", the Pydantic model used to parse/validate that code.
    """

    def wrap(workflow_operation: Callable):
        if instance_info is not None:
            instance_class, instance_model = instance_info
            addReservedName(instance_class) #stop the agent using the instance class name for a group or instance            
            registerInstanceClass(instance_class, instance_model) #record the mapping between the class and its Pydantic model        

        return enwrapWorkflowOperation("meas_config_agent", workflow_operation)
    return wrap

def getUniqueIdx():
    return getWorkflowCallables("meas_config_agent").getUniqueIdx()
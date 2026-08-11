from typing import Literal, Union, List, Optional, Tuple
from typing import Tuple, TypeVar, ClassVar, Callable
import xml.etree.ElementTree as ET
from .hadrons_xml import HadronsXML

from femtomeas.agent_common.common import *
from femtomeas.agent_common.python_update_agent import parameterAgent
from femtomeas.agent_common.callgraph import Node
from .action_config_models import ActionConfig
from .state import State
from .agent_workflows import BaseGroup, BaseGroupHandle, checkValidNewGroupName, getCurrentState
from .agent_workflow_globals import registerWorkflowOperation, getUniqueIdx

def identifyActions(model, group_name:str, state: State):
    role = """identifying all lattice QCD action instances required by the user.

The rules for identifying the required action instances are:
- Create a separate entry for each unique collection of parameters, for example, if the user specifies DWF propagators with Ls=12, M5=1.8 and masses of 0.03 and 0.05, create two separate action instances with different mass values.    
- Create a separate entry for each action instance, even if the action appears multiple times with different parameters.
- Your list must include every action instance explicitly mentioned, and only those. Do not invent instances. do not combine instances unless the user explicitly describes them as the same.
"""

    parameter_rules = [
        "ActionConfig.action: Insert the action type (e.g. DWF, WilsonClover) associated with the instance",
        
        """ActionConfig.name:
  - You must assign a unique tag/name to the instance via the ActionConfig.name field. Do not ask the user to specify a tag.     
  - Never use the same tag for different instances.
  - The tag should include the action name and enough of the parameter values to uniquely distinguish it among the other action instances, prefering shorter tags if possible.""",
        ]
    
    user_info_rules = """Use an empty string""" 
    
    def group_validate(actions):
        for i in range(len(actions)):
            for j in range(i+1, len(actions)):
                if actions[i].action == actions[j].action:
                    return (False, f"Action instances {actions[i].name} and {actions[j].name} have the same parameters. Action instances must be unique.")
        return (True, "")

    inst = state.getInstanceCode("actions")
    
    updated_action_code, group_action_code, _ = parameterAgent(model, ActionConfig, "actions", inst.value, group_name, None, role, tools=[], \
                                                           tool_rules=[], parameter_rules=parameter_rules, user_info_rules=user_info_rules, group_validator=group_validate)

    inst.value = updated_action_code
    return group_action_code


class ActionGroupHandle(BaseGroupHandle):
    pass

class ActionGroup(BaseGroup):
    handle_type : ClassVar[type] = ActionGroupHandle
    code: str = Field(..., description="Code for generating the list of action instances in the group")

@registerWorkflowOperation(instance_info=("actions", ActionConfig) )
def createActionGroup(group_name: str)->ActionGroupHandle:
    def doit(group_name: str):
        state, llm_model = getCurrentState()
        #agent builds a group, adding new action instances as needed
        assert group_name not in state.groups
        state.groups[group_name] = ActionGroup(code = identifyActions(llm_model, group_name, state )    ) 
        print("createActionGroup: ", state.groups[group_name].code,  "\nActions is now: ", state.instances["actions"])   
        return group_name

    checkValidNewGroupName(group_name)
    return ActionGroupHandle(group_name, Node(f"createActionGroup_{getUniqueIdx()}", lambda: doit(group_name) ) )


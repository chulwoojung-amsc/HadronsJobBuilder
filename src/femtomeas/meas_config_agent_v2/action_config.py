from langchain_core.messages import BaseMessage
from langchain.messages import (
    SystemMessage,
    HumanMessage,
    ToolCall,
    AIMessage
)

from typing import Literal, Union, List, Optional, Tuple
import xml.etree.ElementTree as ET
from femtomeas.meas_config_agent.hadrons_xml import HadronsXML

from femtomeas.agent_common.common import *
from femtomeas.agent_common.python_update_agent import parameterAgent
from .action_config_models import ActionConfig
from .state import State

def identifyActions(model, observable_tag:str, observable_info:str, state: State):
    role = "identifying all lattice QCD action instances required to compute the observable, based solely on user input."

    parameter_rules = [
        """actions:
  Perform the following workflow:
    1) If the user has not already done so in the input, ask the user to specify which action type or types they want to use. Typically a single action type is used for all propagators so you should phrase your question as if they will select just one of the options, but explain in parentheses that they are able to choose different actions if desired. In this question, *do not* list the parameters associated with those action types.
    2) Identify the set of action instances required for the observable according to the rules below.
    3) Follow your instructions for writing the appropriate code for the ActionConfig instances.

  The rules for identifying the required action instances are:
  - Create a separate entry for each unique collection of parameters, for example, if the user specifies DWF propagators with Ls=12, M5=1.8 and masses of 0.03 and 0.05, create two separate action instances with different mass values.    
  - Create a separate entry for each action instance, even if the action appears multiple times with different parameters.
  - Your list must include every action instance explicitly mentioned, and only those. Do not invent instances. do not combine instances unless the user explicitly describes them as the same.""",
        
        "ActionConfig.action: Insert the action type (e.g. DWF, WilsonClover) associated with the instance",
        
        """ActionConfig.name:
  - You must assign a unique tag/name to the instance via the ActionConfig.name field. Do not ask the user to specify a tag.     
  - Never use the same tag for different instances.
  - The tag should include the action name and enough of the parameter values to uniquely distinguish it among the other action instances, prefering shorter tags if possible.""",
        ]
    
    user_info_rules = """- Summarize any information relevant to what observables/propagators this action will be used for provided by the user. Do not ask the user to provide this summary.
- It is important that any positional information about the propagator be included, for example whether it is the first or second propagator of a two-point function, or if it is a 'spectator' quark in a baryon.
- If the user does not specify any details, use an empty string. For example, if the user specifies that this action will be used for light quark propagators, enter "use for all light quark propagators" in user_info.""" 

    obs_type = state.getObservableType(observable_tag)
    info = ""
    if observable_info != "":
        info = f"The following information is known about the actions for this observable:\n{observable_info}"

    instructions = f"""Perform your workflow for the observable {observable_tag} with type:
{obs_type.model_dump_json()}
{info}

The following knowledge applies to this observable:
{obs_type.skill()}
"""

    def group_validate(actions):
        for i in range(len(actions)):
            for j in range(i+1, len(actions)):
                if actions[i].action == actions[j].action:
                    return (False, f"Action instances {actions[i].name} and {actions[j].name} have the same parameters. Action instances must be unique.")
        return (True, "")

    print("INSTRUCTIONS\n", instructions)
    input_obs_action_code = None
    if state.observable_actions is not None and observable_tag in state.observable_actions:
        input_obs_action_code = state.observable_actions[observable_tag]

    updated_action_code, obs_action_code, invalidate_later_workflow_stages = parameterAgent(model, ActionConfig, "actions", state.actions, f"{observable_tag}_actions", input_obs_action_code, role, tools=[], \
                                                           tool_rules=[], parameter_rules=parameter_rules, user_info_rules=user_info_rules, input_messages=[ HumanMessage(instructions) ], group_validator=group_validate)

    state.actions = updated_action_code
    if state.observable_actions is None:
        state.observable_actions = dict()
    state.observable_actions[observable_tag] = obs_action_code

    return invalidate_later_workflow_stages
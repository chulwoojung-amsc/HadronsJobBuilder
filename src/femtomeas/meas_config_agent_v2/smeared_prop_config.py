from langchain_core.messages import BaseMessage
from langchain.messages import (
    SystemMessage,
    HumanMessage,
    ToolCall,
    AIMessage
)

from femtomeas.meas_config_agent.hadrons_xml import HadronsXML
from femtomeas.agent_common.common import *
from femtomeas.agent_common.python_update_agent import parameterAgent
from .smeared_prop_config_models import SmearedPropagatorConfig
from .state import State

def identifySmearedPropagators(model, observable_tag:str, observable_info:str, state: State):
    role = """creating configurations for each smeared propagator required by the user.

    Previous agent interactions have identified a set of observables and their required propagators. Propagators may be smeared at their sink location prior to being contracted. You must create SmearedPropagatorConfig instances for each required smeared propagator.     

    ---------------------------------------
    Rules for smeared propagator instances
    ---------------------------------------
    - Create a separate instance for each unique collection of smeared propagator parameters, for example if the user wall-smeared propagator with two different momenta, create two separate SmearedPropagatorConfig instances with each momenta.    
    - Your list must include every smeared propagator instance explicitly mentioned, and only those. Do not invent instances. Do not combine instances unless the user explicitly describes them as the same.
    - Only create separate entries for smeared propagators whose parameters differ, even if those smeared propagators will be associated with different observables.,
    """

    parameter_rules = [
       """SmearedPropagatorConfig.name:
  - You must assign a unique tag/name to the instance via the SmearedPropagator.name field. Do not ask the user to specify a tag.
  - Never use the same tag for different instances.
  - The tag should include the smearing type and enough of the parameter values to uniquely distinguish it among the other instances, prefering shorter tags if possible."""
         ]
            
    user_info_rules = """- For the 'user_info' field, you must summarize any information relevant to what observables this smeared propagator will be used for provided by the user. Never ask the user to provide this parameter.
- For observables with more than one propagator argument, you must record for which propagator the smeared propagator is to be used.
- If the user does now specify any details, use an empty string."""
    
    #Generate instructions
    ########################
    obs_type = state.getObservableType(observable_tag)
    info = ""
    if observable_info != "":
        info = f"The following information is known about the smeared propagators for this observable:\n{observable_info}"

    instructions = f"""Perform your workflow for the observable {observable_tag} with type:
{obs_type.model_dump_json()}
{info}

The following knowledge applies to this observable:
{obs_type.skill()}

The complete set of propagator instances are generated using the following code:
{state.propagators}

For this observable you must use the propagator instances described by the following code:
{state.observable_propagators[observable_tag]}
"""
    
    print("INSTRUCTIONS\n", instructions)

    ##############################

    def checkAll(sprops):
        print("SMEARED PROP CHECK",type(sprops),len(sprops),type(sprops[0]) if len(sprops) > 0 else None)
        val = True
        reason = ""
        
        for i in range(len(sprops)):
            p = sprops[i].check(state)
            if not p[0]:
                val=False
                reason += f"\nsprops[{i}] ({sprops[i].name}): {p[1]}"
        return (val,reason)

    input_obs_sprop_code = state.observable_smeared_propagators[observable_tag] if state.observable_smeared_propagators is not None and observable_tag in state.observable_smeared_propagators else None

    updated_sprop_code, obs_sprop_code, invalidate_later_workflow_stages = parameterAgent(model, SmearedPropagatorConfig, "smeared_propagators", state.smeared_propagators, f"{observable_tag}_smeared_propagators", \
                                                                                          input_obs_sprop_code, role, tools=[], input_messages=[ HumanMessage(instructions) ], parameter_rules=parameter_rules, user_info_rules=user_info_rules, group_validator=checkAll )        

    state.smeared_propagators = updated_sprop_code

    if state.observable_smeared_propagators is None:
        state.observable_smeared_propagators = dict()
    state.observable_smeared_propagators[observable_tag] = obs_sprop_code

    return invalidate_later_workflow_stages
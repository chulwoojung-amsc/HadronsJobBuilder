from langchain_core.messages import BaseMessage
from langchain.messages import (
    SystemMessage,
    HumanMessage,
    ToolCall,
    AIMessage
)
from typing import Literal, Union, List, Optional, Tuple
from femtomeas.agent_common.common import *
from femtomeas.agent_common.python_update_agent import parameterAgent
from femtomeas.meas_config_agent.meas_agent_common import Gammas
from .observable_config_models import ObservableConfig, getMesonGammas, mesonSpecialKeywords
from .state import State


@tool
def getMesonGammasTool(op : mesonSpecialKeywords)-> List[Gammas]:
    """Get the list of Gammas (combination of Gamma-matrices in the Euclidean Clifford algebra) associated with a particular meson operator in the list of special meson keywords"""
    return getMesonGammas(op)

def configureMeson2pt(model, observable_tag:str, observable_info:str, state: State):
    observable_type = "meson2pt_config"
    
    role = f"""for building a list of lattice QCD observable instances and their associated parameters for an observable set with observable type '{observable_type}'. For this set you must instantiate the required number of observable instances and determine their propagators and other parameters."""

    parameter_rules = ["""observable_configs:

  Perform the following workflow:
    1. Parse the user information and background knowledge for the observable set
    2. Determine the ObservableConfig instances required to compute all observable types specified by the user. Follow the rules below.
    3. Instantiate the ObservableConfig instances, populate their parameters and add them to 'observable_configs',

  Rules for ObservableConfig instances:
  - A separate ObservableConfig is required for each unique combination of parameters other than propagators (these are treated separately using the "instances" parameter), even if the observable class is the same. For example, if the user wants to compute the pion and vector two-point functions, create two instances of ObservableConfig, one for the pion and one for the vector. 
  - Your list must include every observable in the list and only those. Do not invent observables, do not combine observables, and do not add details that are not explicitly provided by the user.
  - Do not invent or infer any information not explicitly obtained from the message history.
  - The 'user_info' fields of the propagators and smeared propagators may contain instructions on how to use specific instances; if so, follow those instructions. If not, instantiate ObservableConfig instances for all valid combinations of propagators.
  - You must follow the instructions in the 'user_info' field of *both* the propagators and smeared propagators. This may require instantiating multiple ObservableConfig instances to satisfy both sets of instructions.
                      
  - Create a different instance for each unique combination of propagators
  - Treat the instructions for smeared and unsmeared propagators as applying to separate instances of Meson2ptConfig. 
""",  

    """Meson2ptInstance.name:
  - You must assign a unique tag/name to the instance. Do not ask the user for this parameter
  - Never use the same tag for different instances.
  - The tag should include the observable type and enough of the parameter values to uniquely distinguish it among the other instances, prefering shorter tags if possible.""",
                       
    """Meson2ptInstance.propagators:
  - Use the message history and the skills descriptions of the observables to identify the propagators required to compute this observable and note their 'name' fields. Use only the names of propagators, not of other types of instance (e.g. sources, solvers, actions)
  - Do not invent names for propagators, use only those assigned to existing propagators in your message history.
  - You MUST ensure the order of the two propagators in the Tuple matches the role of the two quarks. For example, if the user says "prop_1" should be used as the first (or incoming) quark, ensure it is the first entry.
  - You can use either two smeared propagators or two unsmeared propagators, but you cannot mix smeared and unsmeared propagators
  """,

    """Meson2ptInstance.sink:
  - When using smeared propagators, you must use a ContractionSinkNone for this parameter
  - When using regular/unsmeared propagators, you must choose the sink type according to the user's input. The most common sink type is ContractionSinkPoint (point sink); you may suggest this to the user.""",
                       
    """Meson2ptConfig.sink_gammas and Meson2ptConfig.source_gammas:
  - These are lists of Gamma-matrix combinations for the sink and source locations, respectively.
  - If the user has not previously provided these parameters, perform the following workflow:
    1) Based upon the observable type and other user-provided information, attempt to identify the special "mesonSpecialKeywords" keywords that describe the meson states at the source and sink. You can use the same keyword for the source and sink mesons unless the user has specified otherwise.

    2) - If you are able to identify the keywords, you must describe the keywords you identified for both source and sink to the user, and ask them to confirm. Ensure you specify both source and sink keywords even if they are the same. Do not ask more than one question at a time. You can ask the user to confirm multiple keywords at once, but only in the form of a single question.
       - If you are *not* able to identify the keywords, ask the user to either choose the keywords or else manually specify the lists of Gamma-matrix combinations at the source and sink

    3) - If you have identified or have been given the keywords, you must call the getMesonGammasTool tool to obtain the list of Gamma-matrix combinations for the source and sink. 
       - Otherwise, if the user specified the Gamma-matrix combinations, use those to populate sink_gammas and source_gammas and finish this workflow
       - Never guess the gamma matrix combinations
       - These lists must always contain one or more Gamma-matrix combination; they can never be empty."""
                       ]

    tools = [getMesonGammasTool]
    tool_rules = []
    
    user_info_rules = "-No user_info is required for this stage."

    #Generate instructions
    ########################
    obs_type = state.getObservableType(observable_tag)
    info = ""
    if observable_info != "":
        info = f"The following information is known about the observable instances required for this observable set:\n{observable_info}"

    instructions = f"""Perform your workflow for the observable set {observable_tag} with type:
{obs_type.model_dump_json()}
{info}

The following knowledge applies to this observable:
{obs_type.skill()}

The complete set of propagator instances are generated using the following code:
{state.propagators}

The complete set of smeared propagator instances are generated using the following code:
{state.smeared_propagators}

For this observable you must use the propagator instances described by the following code:
{state.observable_propagators[observable_tag] if state.observable_propagators is not None and observable_tag in state.observable_propagators else "props = []"}

For this observable you must use the smeared propagator instances described by the following code:
{state.observable_smeared_propagators[observable_tag] if state.observable_smeared_propagators is not None and observable_tag in state.observable_smeared_propagators else "smeared_props = []"}
"""
    
    print("INSTRUCTIONS\n", instructions)

    ##############################

    def instanceCheck(obs: ObservableConfig):
        if obs.obs.type != observable_type:
            return (False, f"Observable 'type' must be {observable_type}")
        
        return obs.check(state)

    input_obs_code = None
    if state.observable_observable_configs is not None and observable_tag in state.observable_observable_configs:
        input_obs_code = state.observable_observable_configs[observable_tag]

    updated_obs_code, obs_obs_code, _ = parameterAgent(model, ObservableConfig, "observable_configs", state.observable_configs, f"{observable_tag}_observables", input_obs_code,\
                                                        role, tools=tools, tool_rules=tool_rules, parameter_rules=parameter_rules, input_messages=[ HumanMessage(instructions) ], instance_validator=instanceCheck, user_info_rules=user_info_rules)

    state.observable_configs = updated_obs_code

    if state.observable_observable_configs is None:
        state.observable_observable_configs = dict()
    state.observable_observable_configs[observable_tag] = obs_obs_code
from langchain_core.messages import BaseMessage
from langchain.messages import (
    SystemMessage,
    HumanMessage,
    ToolCall,
    AIMessage
)

from pydantic import BaseModel, Field, ConfigDict, NonNegativeInt, TypeAdapter
from typing import Literal, Union, List, Optional, Tuple
from langchain.agents.structured_output import ToolStrategy, ProviderStrategy
from langchain.agents import create_agent
from femtomeas.meas_config_agent.hadrons_xml import HadronsXML
import json
from femtomeas.agent_common.common import *
from femtomeas.agent_common.python_update_agent import parameterAgent
from femtomeas.meas_config_agent.meas_agent_common import Gammas
from .source_config_models import SourceConfig
from .state import State

def identifySources(model, observable_tag:str, observable_info:str, state: State):
    role = """creating instances of SourceConfig for every propagator source required to compute the observable.

    Sources are inputs to constructing quark propagators. A source instance has a source type (e.g. point, wall) along with a set of parameters that depend on the source type. Each propagator requires a source, but can share the same source instance.

    Perform the following workflow:
    1) Check the message history to see if the source types of the required propagators has been specified.

       If the source types have not yet been specified:
         - ask the user to specify what source *types* they wish to use for which propagators. This question should not be specific to one observable or propagator; rather you should allow the user the freedom to specify information that could apply to multiple or even all propagators. In your question, list the sources that you support but do not list their associated parameters.
           For example, "Specify the sources required for the calculation (supported options: <OPTIONS>)."
         - Do not ask the user to provide parameters at this stage.
          

       If the source types have been specified:
         - indicate to the user what source types you have identified and ask them if this is correct.
         - take note of any additional information about the parameters of those sources that is contained in the message history, e.g. source timeslices or locations 
    2) Identify the set of source instances required for the observable according to the rules below.
    3) Follow your instructions for writing the appropriate code for the SourceConfig instances.
        
  The rules for identifying the required source instances are as follows:
  - Create a separate entry for each unique collection of source parameters, for example if the user specified propagators with point sources at [0,0,0,0] and [12,24,12,24], create two separate source instances with different source locations.
  - Create a separate entry for each source instance, even if the same source type appears multiple times with different parameters.
  - Your list must include every source instance explicitly mentioned, and only those. Do not invent instances. Do not combine instances unless the user explicitly describes them as the same.
  - Only create separate entries for sources whose source parameters differ, even if those sources will be associated with different actions in their associated propagators. For instance, if there are two action instances, 'action_a' and 'action_b' which both need wall sources with t=0, create only one wall source instance.

  Notes:
  - Do not confuse sink smearing and sources. Sink smearing is performed on the solutions of inverting the Dirac matrix upon a source, and is entirely independent from the form of the source.
  - Do not describe non-local sources as "smeared" sources.
  - When the user asks for a point source, assume that they mean PointSource and NOT GaussianSource unless they specifically mention the gaussian source
    """

    parameter_rules = [
       """SourceConfig.name:
  - You must assign a unique tag/name to the instance via the SourceConfig.name field. Do not ask the user to specify a tag.
  - Never use the same tag for different instances.
  - The tag should include the action name and enough of the parameter values to uniquely distinguish it among the other source instances, prefering shorter tags if possible.""",
      
      """SourceConfig.source:
  - Insert the appropriate schema for the source type""",

      """SeqGammaSource.q, SeqGammaSource.q_prop_info, SeqGammaSource.q_source_name:
  For these parameters you MUST perform the following workflow:
    1) Explain to the user that they can provide a specific propagator name that will be used as a handle, or else they can let the agent decide on the name. Tell them that they can provide other relevant information that will allow them to distinguish between different input propagators.
    2) If the user specified a propagator name, record it as the "q" parameter. If they let you decide, you must choose a new, unique name for that propagator and record it instead. The name you choose must differ from the name of the sequential source or any other source.
    3) - If the user has previously specified which source to use for the input propagator "q", you MUST confirm this with the user. If accepted, use the name of this source as the "q_source_name" 
       - If the user has NOT previously specified the source or rejected your suggestion you MUST ask the user to specify what source will be used for the input propagator "q".
        - Based on their response, determine if this source already exists,
           - if so use the name of that source for q_source_name
           - if not, instantiate a new SourceConfig instance in the "sources" field of your output for this extra source and obtain its parameters from the user
    4) In the "q_prop_info" field, record any other information provided by the user about the propagator such as its mass, action, etc. If no further information is given by the user, do not prompt them to do so again; instead use an empty string for "q_prop_info".""",

      """SeqGammaSource.t_a, SeqGammaSource.t_b: Do not ask for these parameters separately. Instead, explain that t_a and t_b denote the start and end timeslices of the sequential source, and that they can be equal. Ask for both parameters to be specified together."""
        
    ]
    
    additional_user_query_rules = [    
        """"Do not require a specific format for the user's input. For example, never ask "provide each value separated by a comma" """,
        "When asking for parameters, phrase your questions to refer to groups of propagators that share the same partial set of parameters rather than specific propagators.",
        "Do not assume that the sources associated with propagators in these groups will all have the same parameters.",
        "Do not include options in your questions that accept a single parameter value for all propagators in the group. If the user wants to specify a parameter that applies to more than one propagator in the group they will do so explicitly.",
        """Use plurals for parameter names associated with groups containing more than one propagator
  Examples:
    "Provide the timeslices for the wall sources." (plural)
    "What source locations are associated with the three baryon point source propagators?" (plural)
    "What is the momentum for the wall-momentum source?" (singular)
    "What timeslices should be used for the wall sources used for the pion and vector two-point functions?" (plural)
  Use the following plurals:
    timeslice -> time slices
    location -> locations
    momentum -> momenta""",
    
        "When asking a question referring to a group, ensure your question clearly identifies the group."
        ]
    
    def checkAll(sources):
        print("SOURCE CHECK",type(sources),len(sources),type(sources[0]) if len(sources) > 0 else None)
        src_names = [s.name for s in sources]
        val = True
        reason = ""
        for i in range(len(sources)):
            for j in range(i+1, len(sources)):
                if sources[i].source == sources[j].source:
                    val=False
                    reason += f"\nSource {sources[i].name} and {sources[j].name} are the same. Sources must be unique"
        
        for i in range(len(sources)):
            p = sources[i].check(state, src_names)
            if not p[0]:
                val=False
                reason += f"\nsources[{i}] ({sources[i].name}): {p[1]}"
        return (val,reason)


    user_info_rules = """- For the 'user_info' field, you must summarize any information relevant to what observables/propagators this source will be used for provided by the user. Never ask the user to provide this parameter.
- It is important that any positional information about the propagator be included, for example whether it is the first or second propagator of a two-point function, or if it is a 'spectator' quark in a baryon.
- If the user does now specify any details, use an empty string. For example, if the user specifies that this source will be used for light quark propagators, enter "use for all light quark propagators" in user_info."""


    obs_type = state.getObservableType(observable_tag)
    info = ""
    if observable_info != "":
        info = f"The following information is known about the sources for this observable:\n{observable_info}"

    ###Generate instructions
    instructions = f"""Perform your workflow for the observable {observable_tag} with type:
{obs_type.model_dump_json()}
{info}

The following knowledge applies to this observable:
{obs_type.skill()}
"""

    print("INSTRUCTIONS\n", instructions)
    #########################################

    input_obs_source_code = state.observable_sources[observable_tag] if state.observable_sources is not None and observable_tag in state.observable_sources else None
    
    updated_source_code, obs_source_code, invalidate_later_workflow_stages = parameterAgent(model, SourceConfig, "sources", state.sources, f"{observable_tag}_sources", input_obs_source_code, role, \
                                                                                            tools=[], input_messages=[ HumanMessage(instructions) ], parameter_rules=parameter_rules, group_validator=checkAll, additional_user_query_rules=additional_user_query_rules, user_info_rules=user_info_rules )
    state.sources = updated_source_code

    if state.observable_sources is None:
        state.observable_sources = dict()
    state.observable_sources[observable_tag] = obs_source_code

    return invalidate_later_workflow_stages

from pydantic import BaseModel, Field, ConfigDict, NonNegativeInt, TypeAdapter
from typing import Literal, Union, List, Optional, Tuple, ClassVar
from femtomeas.meas_config_agent.hadrons_xml import HadronsXML
import json
from femtomeas.agent_common.common import *
from femtomeas.agent_common.python_update_agent import parameterAgent
from femtomeas.meas_config_agent.meas_agent_common import Gammas
from femtomeas.meas_config_agent_v2.source_config_models import SourceConfig, SeqGammaSource
from .state import State
from .agent_workflows import BaseGroup, BaseGroupHandle, checkValidNewGroupName, getCurrentState
from .agent_workflow_globals import registerWorkflowOperation, getUniqueIdx
from femtomeas.agent_common.callgraph import Node

def identifySources(model, group_name : str, state : State):
    role = """creating instances of SourceConfig for every propagator source required by the user.

    Sources are inputs to constructing quark propagators. A source instance has a source type (e.g. point, wall) along with a set of parameters that depend on the source type.
        
  The rules for identifying the required source instances are as follows:
  - Create a separate entry for each unique collection of source parameters, for example if the user specified  point sources at [0,0,0,0] and [12,24,12,24], create two separate source instances with different source locations.
  - Create a separate entry for each source instance, even if the same source type appears multiple times with different parameters.
  - Your list must include every source instance explicitly mentioned, and only those. Do not invent instances. Do not combine instances unless the user explicitly describes them as the same.
  - Only create separate entries for sources whose source parameters differ, even if those sources will be associated with different actions in their associated propagators. For instance, if there are two action instances, 'action_a' and 'action_b' which both need wall sources with t=0, create only one wall source instance.

  Notes:
  - Do not confuse sink smearing and sources. Sink smearing is performed on the solutions of inverting the Dirac matrix upon a source, and is entirely independent from the form of the source.
  - Do not describe non-local sources as "smeared" sources.
  - When the user asks for a point source, assume that they mean PointSource and NOT GaussianSource unless they specifically mention the gaussian source
  - You do not support sequential sources (SeqGammaSource)
    """

    parameter_rules = [
       """SourceConfig.name:
  - You must assign a unique tag/name to the instance via the SourceConfig.name field. Do not ask the user to specify a tag.
  - Never use the same tag for different instances.
  - The tag should include the action name and enough of the parameter values to uniquely distinguish it among the other source instances, prefering shorter tags if possible.""",
      
      """SourceConfig.source:
  - Insert the appropriate schema for the source type"""
    ]

#TODO: Sequential sources should be separate
#       """SeqGammaSource.q, SeqGammaSource.q_prop_info, SeqGammaSource.q_source_name:
#   For these parameters you MUST perform the following workflow:
#     1) Explain to the user that they can provide a specific propagator name that will be used as a handle, or else they can let the agent decide on the name. Tell them that they can provide other relevant information that will allow them to distinguish between different input propagators.
#     2) If the user specified a propagator name, record it as the "q" parameter. If they let you decide, you must choose a new, unique name for that propagator and record it instead. The name you choose must differ from the name of the sequential source or any other source.
#     3) - If the user has previously specified which source to use for the input propagator "q", you MUST confirm this with the user. If accepted, use the name of this source as the "q_source_name" 
#        - If the user has NOT previously specified the source or rejected your suggestion you MUST ask the user to specify what source will be used for the input propagator "q".
#         - Based on their response, determine if this source already exists,
#            - if so use the name of that source for q_source_name
#            - if not, instantiate a new SourceConfig instance in the "sources" field of your output for this extra source and obtain its parameters from the user
#     4) In the "q_prop_info" field, record any other information provided by the user about the propagator such as its mass, action, etc. If no further information is given by the user, do not prompt them to do so again; instead use an empty string for "q_prop_info".""",

#       """SeqGammaSource.t_a, SeqGammaSource.t_b: Do not ask for these parameters separately. Instead, explain that t_a and t_b denote the start and end timeslices of the sequential source, and that they can be equal. Ask for both parameters to be specified together."""


    additional_user_query_rules = [    
        """"Do not require a specific format for the user's input. For example, never ask "provide each value separated by a comma" """
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

    def instanceCheck(source):
        if isinstance(source.source, SeqGammaSource):
            return (False, "You do not support SeqGammaSource")
        return (True, "")

    user_info_rules = """- You must use an empty string for this field."""

    inst = state.getInstanceCode("sources")
    
    updated_source_code, group_source_code, _ = parameterAgent(model, SourceConfig, "sources", inst.value, group_name, None, role, \
                                                            tools=[], parameter_rules=parameter_rules, group_validator=checkAll, instance_validator=instanceCheck,  additional_user_query_rules=additional_user_query_rules, user_info_rules=user_info_rules )
    inst.value = updated_source_code
    return group_source_code


class SourceGroupHandle(BaseGroupHandle):
    pass

class SourceGroup(BaseGroup):
    handle_type : ClassVar[type] = SourceGroupHandle
    code: str = Field(..., description="Code for generating the list of source instances in the group")

@registerWorkflowOperation(instance_info=("sources", SourceConfig) )     
def createSourceGroup(group_name: str)->SourceGroupHandle:
    def doit(group_name: str):
        state, llm_model = getCurrentState()
        #agent builds a group, adding new source instances as needed
        assert group_name not in state.groups
        state.groups[group_name] = SourceGroup(code = identifySources( llm_model, group_name, state ))   
        print("createSourceGroup: ", state.groups[group_name].code,  "\nSources is now: ", state.instances["sources"])    
        return group_name

    checkValidNewGroupName(group_name)
    return SourceGroupHandle(group_name, Node(f"createSourceGroup_{getUniqueIdx()}", lambda: doit(group_name) ) )



from typing import Literal, Union, List, Optional, Tuple, ClassVar
from femtomeas.agent_common.common import *
from femtomeas.agent_common.python_update_agent import parameterAgent, InstanceInfo
from .meas_agent_common import Gammas
from .observable_config_models import ObservableConfig, getMesonGammas, mesonSpecialKeywords, ContractionSinkNone
from femtomeas.agent_common.python_output_agent import executeCodeAndParse
from .state import State
from .agent_workflows import checkValidNewGroupName, getCurrentState, startWorkflowMessage
from .agent_workflow_globals import registerWorkflowOperation, getUniqueIdx
from .agent_workflow_base import BaseRoutingHandle, BaseGroup
from femtomeas.agent_common.callgraph import Node
from .propagator_config import PropagatorGroupHandle, PropagatorGroup
from .smeared_prop_config import SmearedPropagatorGroupHandle, SmearedPropagatorGroup

@tool
def getMesonGammasTool(op : mesonSpecialKeywords)-> List[Gammas]:
    """Get the list of Gammas (combination of Gamma-matrices in the Euclidean Clifford algebra) associated with a particular meson operator in the list of special meson keywords"""
    return getMesonGammas(op)

def configureMeson2pt(model, group_name: str, prop_group_name:str, state: State, extra_info : str):
    prop_group_code = state.groups[prop_group_name].code
    is_smeared = isinstance(state.groups[prop_group_name], SmearedPropagatorGroup)

    #=====================
    if is_smeared:
        prop_info = f"""
  - You are restricted to using smeared propagators from the following subset:
  {prop_group_code}
        
  where the complete set of smeared propagators is defined through the following Python code:
  {state.instances["smeared_propagators"]}

  and the corresponding base propagators are defined through
  {state.instances["propagators"]}
"""
    else:
        prop_info = f""" 
  - You are restricted to using propagators from the following subset:
    {prop_group_code}

    where the complete set of propagators is defined through the following Python code:
    {state.instances["propagators"]}
"""
  #==========================
    
    role = f"""for building a collection of meson two-point function observable instances and their associated parameters based on your conversation with the user.

  Meson two-point functions are typically used to compute particle masses or decay constants (e.g. f_pi).

  This observable requires two propagators, that are contracted together at some timeslice-localized sink. The first propagator argument has quark flow from source to sink, and the second propagator argument has gamma^5-hermiticity applied to it such that the quark flow is from sink to source. We often refer to these propagators by the direction of quark flow into the vertex, i.e. "incoming" for the first quark and "outgoing" for the second.

  Pion and kaon correlators are both pseudoscalar two-point functions, the difference being that the pion usually has both quarks with the same (light) mass, whereas the kaon has one heavy and one light quark.
  {prop_info}
    
  - Ask the user to specify which combinations of propagators to use from within the subset above.

-----------------------------    
ObservableConfig instances rules
-----------------------------
  - A separate ObservableConfig is required for each unique combination of parameters, even if the observable class is the same. For example, if the user wants to compute the pion and vector two-point functions, create two instances of ObservableConfig, one for the pion and one for the vector. 
  - Your list must include every observable in the list and only those. Do not invent observables, do not combine observables, and do not add details that are not explicitly provided by the user.
  - Do not invent or infer any information not explicitly obtained from the message history.  
  - Create a different instance for each unique combination of propagators
    """

    parameter_rules = ["""ObservableConfig.obs must be of type Meson2ptConfig""",

    """Meson2ptInstance.name:
  - You must assign a unique tag/name to the instance. Do not ask the user for this parameter
  - Never use the same tag for different instances.
  - The tag should include the observable type and enough of the parameter values to uniquely distinguish it among the other instances, prefering shorter tags if possible.""",
                       
    """Meson2ptInstance.propagators:
  - Use only the names of propagators from within the list provided above. Do not invent propagator names.
  - You MUST ensure the order of the two propagators in the Tuple matches the role of the two quarks. For example, if the user says "prop_1" should be used as the first (or incoming) quark, ensure it is the first entry.
  """,

    f"""Meson2ptInstance.sink:
  - You must use {"ContractionSinkPoint" if not is_smeared else "ContractionSinkNone"} for this parameter"""
                       
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

    prp, _ = executeCodeAndParse(prop_group_code, InstanceInfo, prop_group_name) 
    used_props = [ a.instance_tag for a in prp ]

    def instanceCheck(obs: ObservableConfig):
        if obs.obs.type != "meson2pt_config":
            return (False, f"Observable 'type' must be 'meson2pt_config'")
        if obs.obs.propagators[0] not in used_props or obs.obs.propagators[1] not in used_props:
            return (False, f"You must only use propagators from within the provided subset")
        if is_smeared and not isinstance(obs.obs.sink, ContractionSinkNone):
            return (False, f"You must use sink=ContractionSinkNone for smeared propagators")
        if not is_smeared and isinstance(obs.obs.sink, ContractionSinkNone):
            return (False, f"You must not use sink=ContractionSinkNone for unsmeared propagators")
        
        return obs.check(state)

    additional_user_query_rules = [
      """When asking for the meson or observable type, you MUST list the meson types you know about (pion, kaon etc) but ALSO mention that the user can directly specify the gamma matrices.""",
      """NEVER insist that the user answer your question in a specific format or ordering."""
    ]

    inst = state.getInstanceCode("observable_configs")
   
    updated_obs_code, obs_group_code, _ = parameterAgent(model, ObservableConfig, "observable_configs", inst.value, group_name, None,\
                                                        role, tools=tools, tool_rules=tool_rules, parameter_rules=parameter_rules, instance_validator=instanceCheck, user_info_rules=user_info_rules, additional_user_query_rules=additional_user_query_rules, input_messages=startWorkflowMessage(extra_info))

    inst.value = updated_obs_code
    return obs_group_code


class ObservableGroupHandle(BaseRoutingHandle):
    pass

class Meson2ptGroup(BaseGroup):
    handle_type : ClassVar[type] = ObservableGroupHandle
    code: str = Field(..., description="Code for generating the list of meson 2pt function instances in the group")

@registerWorkflowOperation(instance_info=("observable_configs", ObservableConfig) )
def createMeson2ptGroup(group_name: str, propagators: PropagatorGroupHandle | SmearedPropagatorGroupHandle, *, user_info: str = "")->ObservableGroupHandle:
    """Create a Meson2ptGroup and return its handle.

user_info: if specified by the user, provide the meson type(s) or other information relevant to creating these observables from the input propagators.

The meson two-point function (aka meson correlator) is used to describe the lattice propagation of a meson such as a pion or kaon        
    """
    checkValidNewGroupName(group_name)
    def doit(group_name, gprops_group_name: str): 
        state, llm_model = getCurrentState()
        assert gprops_group_name in state.groups and isinstance(state.groups[gprops_group_name], (PropagatorGroup, SmearedPropagatorGroup) )
        state.groups[group_name] = Meson2ptGroup(code = configureMeson2pt(llm_model, group_name, gprops_group_name, state, user_info) )
        return group_name
   
    return ObservableGroupHandle(Node(f"createMeson2ptGroup_{getUniqueIdx()}", lambda gprops: doit(group_name, gprops), input_deps=[propagators] ) )



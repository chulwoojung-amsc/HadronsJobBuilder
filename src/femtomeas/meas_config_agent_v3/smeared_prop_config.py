from femtomeas.meas_config_agent.hadrons_xml import HadronsXML
from femtomeas.agent_common.common import *
from femtomeas.agent_common.python_update_agent import parameterAgent, InstanceInfo
from femtomeas.meas_config_agent_v2.smeared_prop_config_models  import SmearedPropagatorConfig
from femtomeas.agent_common.python_output_agent import executeCodeAndParse
from .state import State
from .agent_workflows import BaseGroup, BaseGroupHandle, registerWorkflowOperation, checkValidNewGroupName, getUniqueIdx, addReservedName, getCurrentState
from femtomeas.agent_common.callgraph import Node
from typing import ClassVar
from .propagator_config import PropagatorGroup, PropagatorGroupHandle

def identifySmearedPropagators(model, group_name: str, input_props_group_name:str, state: State):
    input_props_group_code = state.groups[input_props_group_name].code

    role = f"""creating configurations for each smeared propagator required by the user.

    Propagators may be smeared at their sink location prior to being contracted. You must create SmearedPropagatorConfig instances for each smeared propagator required by the user.

    - You are restricted only to smearing propagators in the following subset:
    {input_props_group_code}

    - The complete set of unsmeared input propagators is defined by the following Python code:
    {state.instances["propagators"]}

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
            
    user_info_rules = """- You must use an empty string for this field."""

    prp, _ = executeCodeAndParse(input_props_group_code, InstanceInfo, input_props_group_name) 
    used_props = [ a.instance_tag for a in prp ]
    
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

    def instanceCheck(sprop):
        if sprop.input_prop not in used_props:
            return (False, "Input propagators are restricted to the provided subset")
        return (True,"")
    
    inst = state.getInstanceCode("smeared_propagators")

    updated_sprop_code, group_sprop_code, _ = parameterAgent(model, SmearedPropagatorConfig, "smeared_propagators", inst.value, group_name, None, \
                                                            role, tools=[], parameter_rules=parameter_rules, user_info_rules=user_info_rules, group_validator=checkAll, instance_validator=instanceCheck )        

    inst.value = updated_sprop_code
    return group_sprop_code


class SmearedPropagatorGroupHandle(BaseGroupHandle):
    pass

class SmearedPropagatorGroup(BaseGroup):
    handle_type : ClassVar[type] = PropagatorGroupHandle
    code: str = Field(..., description="Code for generating the list of sink-smeared propagator instances in the group")

    
def createSmearedPropagatorGroup(group_name: str, unsmeared_props: PropagatorGroupHandle)->SmearedPropagatorGroupHandle:
    checkValidNewGroupName(group_name)

    def doit(group_name, gunsmeared_prop_group_name: str):
        state, llm_model = getCurrentState()
        assert gunsmeared_prop_group_name in state.groups and isinstance(state.groups[gunsmeared_prop_group_name], PropagatorGroup)        

        state.groups[group_name] = SmearedPropagatorGroup(code = identifySmearedPropagators(llm_model, group_name, gunsmeared_prop_group_name, state) )
        return group_name
   
    return SmearedPropagatorGroupHandle(group_name, Node(f"createSmearedPropagatorGroup_{getUniqueIdx()}", lambda gunsmeared_props: doit(group_name, gunsmeared_props), input_deps=[unsmeared_props] ) )

registerWorkflowOperation(createSmearedPropagatorGroup)
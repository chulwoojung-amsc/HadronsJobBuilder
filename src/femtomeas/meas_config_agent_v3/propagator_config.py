import json
from pydantic import BaseModel, Field, ConfigDict, NonNegativeInt, TypeAdapter
from typing import Literal, Union, List, Optional, Tuple, ClassVar
from langchain.agents.structured_output import ToolStrategy, ProviderStrategy
from femtomeas.agent_common.common import *
from femtomeas.meas_config_agent.hadrons_xml import HadronsXML
from femtomeas.agent_common.python_output_agent import executeCodeAndParse
from femtomeas.agent_common.python_update_agent import parameterAgent, InstanceInfo
from femtomeas.meas_config_agent_v2.propagator_config_models import PropagatorConfig
from .state import State
from .agent_workflows import BaseGroup, BaseGroupHandle, checkValidNewGroupName, getCurrentState
from .agent_workflow_globals import registerWorkflowOperation, getUniqueIdx
from femtomeas.agent_common.callgraph import Node
from .source_config import SourceGroup, SourceGroupHandle
from .solver_config import SolverGroup, SolverGroupHandle

def identifyPropagators(model, group_name, source_group_name, solver_group_name, state: State): 
    source_group_code = state.groups[source_group_name].code
    solver_group_code = state.groups[solver_group_name].code

    role = f"""identifying the (non-sink-smeared) lattice QCD propagators required by the user and their associated solver and source.

- The sources must be chosen from within the following subset:
{source_group_code}

where the source parameters are defined through the following Python code:
{state.instances["sources"]}

- The solvers must be chosen from within the following subset:
{solver_group_code}

where the solver parameters are defined through the following Python code:
{state.instances["solvers"]}

- Ask the user to specify which combinations of source and solver they wish to generate propagator instances for.

-------------------------
Propagator instance rules
-------------------------
- You must reuse propagator instances that share the same source and solver
"""    

    user_info_rules = """- You must use an empty string for this field."""

    sourc, _ = executeCodeAndParse(source_group_code, InstanceInfo, source_group_name) 
    used_sources = [ a.instance_tag for a in sourc ]

    solv, _ = executeCodeAndParse(solver_group_code, InstanceInfo, solver_group_name) 
    used_solvers = [ a.instance_tag for a in solv ]


    def instanceCheck(prop):
        if not state.isValidInstance(prop.source, "sources"):
            return (False, f"\n-Source instance '{prop.source}' does not exist")        
        if not state.isValidInstance(prop.solver, "solvers"):
            return (False, f"\n-Solver instance '{prop.solver}' does not exist")
        if prop.source not in used_sources:
            return (False, f"\n-Source instance '{prop.source}' is not within the subset source instances associated with this group")
        if prop.solver not in used_solvers:
            return (False, f"\n-Solver instance '{prop.solver}' is not within the subset solver instances associated with this group")
                
        return (True, "")

    def groupCheck(props):
        names = []
        valid = True
        invalid_why = ""
        for i in range(len(props)):
            for j in range(i+1, len(props)):
                if props[i].source == props[j].source and props[i].solver == props[j].solver:
                    invalid_why += f"\n-Propagators '{props[i].name}' and '{props[j].name}' are the same. Propagators must be unique."

        for r in props:
            if r.name in names:
                invalid_why += f"\n-Propagator name '{r.name}' is not unique"
                valid = False
            names.append(r.name)
        return (valid, invalid_why)

    parameter_rules = [
      """PropagatorConfig.name: You must choose a unique tag/name to the propagator instance. Assign this automatically, never ask the user (although they may choose to suggest names if they desire).
  - Never use the same tag for different instances.
  - The tag should include the source and solver names.""",

      ]

    additional_user_query_rules = [        
    "Do not demand a specific format for the user's response"
    ]

    ##############################
    inst = state.getInstanceCode("propagators")

    updated_prop_code, group_prop_code, _ = parameterAgent(model, PropagatorConfig, "propagators", inst.value, group_name, None, role, parameter_rules=parameter_rules, tools=[],
                                                        user_info_rules = user_info_rules, instance_validator=instanceCheck, group_validator=groupCheck, additional_user_query_rules=additional_user_query_rules)

    inst.value = updated_prop_code
    return group_prop_code



class PropagatorGroupHandle(BaseGroupHandle):
    pass

class PropagatorGroup(BaseGroup):
    handle_type : ClassVar[type] = PropagatorGroupHandle
    code: str = Field(..., description="Code for generating the list of propagator instances in the group")

@registerWorkflowOperation(instance_info=("propagators", PropagatorConfig) )    
def createPropagatorGroup(group_name: str, sources: SourceGroupHandle, solvers: SolverGroupHandle)->PropagatorGroupHandle:
    checkValidNewGroupName(group_name)

    def doit(group_name, gsources_group_name: str, gsolvers_group_name: str):
        state, llm_model = getCurrentState()
        assert gsources_group_name in state.groups and isinstance(state.groups[gsources_group_name], SourceGroup)
        assert gsolvers_group_name in state.groups and isinstance(state.groups[gsolvers_group_name], SolverGroup)

        state.groups[group_name] = PropagatorGroup(code = identifyPropagators(llm_model, group_name, gsources_group_name, gsolvers_group_name, state) )
        return group_name
   
    return PropagatorGroupHandle(group_name, Node(f"createPropagatorGroup_{getUniqueIdx()}", lambda gsources, gsolvers: doit(group_name, gsources, gsolvers), input_deps=[sources,solvers] ) )



from langchain_core.messages import BaseMessage
from langchain.messages import (
    SystemMessage,
    HumanMessage,
    ToolCall,
    AIMessage
)
import json
from pydantic import BaseModel, Field, ConfigDict, NonNegativeInt, TypeAdapter
from typing import Literal, Union, List, Optional, Tuple
from langchain.agents.structured_output import ToolStrategy, ProviderStrategy
from femtomeas.agent_common.common import *
from femtomeas.meas_config_agent.hadrons_xml import HadronsXML
from femtomeas.agent_common.python_output_agent import executeCodeAndParse
from femtomeas.agent_common.python_update_agent import parameterModelCall, InstanceInfo
from .propagator_config_models import PropagatorConfig
from .state import State

def identifyPropagators(model, observable_tag:str, observable_info:str, state: State): 
    role = """
You are responsible for identifying the (non-sink-smeared) lattice QCD propagators for the calculation of the observable and their associated solver and source. (Sink-smeared propagators are built from unsmeared propagators and will be created later.)

A propagator instance has a 'source' and 'solver' field that must be set, respectively, to the name of one of the source and solver instances identified previously.

First, identify the set of required propagators by noting how many propagators are required by this observable other relevant information. Then, consider the message history to identify the source and solver combination that uniquely specifies each of the propagators needed for this observable. Also include propagators defined as inputs to sequential sources. Do not specify more propagators than are required for the observable.

If the observable requires a propagator with the same source/solver combination as an existing one, you must re-use the propagator; do not create more propagators than needed.
    
- For each required propagator:
1. Identify the name of the associated source instance and use it to fill the 'source' parameter.
2. Identify the name of the associated solver instance and use it to fill the 'solver' parameter. To perform this identification, combine the parameters of the solver instance with those of the associated action instance, which is tagged by the field 'action'.
3. Assign a unique tag/name to the propagator instance. Never use the same tag for different instances. The tag should include the solver and source name.
    
Propagator instance rules:    
- Your list must include every propagator instance required for the observables, and only those. Do not invent instances.
- You must reuse propagator instances that share the same source and solver
- Do not create separate non-sink-smeared propagators for different sink-smeared propagators. A single non-sink-smeared propagator can be sink smeared in arbitrary ways.
"""    

    user_info_rules = """- For the 'user_info' field, summarize any information relevant to what observables this solver will be used for provided by the user. It is important that any positional information about the propagator be included, for example whether it is the first or second propagator of a two-point function, or if it is a 'spectator' quark in a baryon. If the user does now specify any details, use an empty string. For example, if the user specifies that this propagator will be used for both quarks of the pion two-point function, enter "use for both quarks of the pion two-point function" in user_info."""

    sourc, _ = executeCodeAndParse(state.observable_sources[observable_tag], InstanceInfo, f"{observable_tag}_sources") 
    used_sources = [ a.instance_tag for a in sourc ]

    solv, _ = executeCodeAndParse(state.observable_solvers[observable_tag], InstanceInfo, f"{observable_tag}_solvers") 
    used_solvers = [ a.instance_tag for a in solv ]
   
    def instanceCheck(prop):
        if not state.isValidSource(prop.source):
            return (False, f"\n-Source instance '{prop.source}' does not exist")        
        if not state.isValidSolver(prop.solver):
            return (False, f"\n-Solver instance '{prop.solver}' does not exist")
        if not prop.source in used_sources:
            return (False, f"\n-Source instance '{prop.source}' is not within the list of used source instances for this observable")        
        if not prop.solver in used_solvers:
            return (False, f"\n-Solver instance '{prop.solver}' is not within the list of used solver instances for this observable")        
                
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

    input_obs_prop_code = state.observable_propagators[observable_tag] if state.observable_propagators is not None and observable_tag in state.observable_propagators else None

    #Generate instructions
    ########################
    obs_type = state.getObservableType(observable_tag)
    info = ""
    if observable_info != "":
        info = f"The following information is known about the propagators for this observable:\n{observable_info}"

    instructions = f"""Perform your workflow for the observable {observable_tag} with type:
{obs_type.model_dump_json()}
{info}

The following knowledge applies to this observable:
{obs_type.skill()}

The complete set of source instances are generated using the following code:
{state.sources}

The complete set of solver instances are generated using the following code:
{state.solvers}

The complete set of action instances are generated using the following code:
{state.actions}

For this observable you must use the source instances described by the following code:
{state.observable_sources[observable_tag]}

For this observable you must use the solver instances described by the following code:
{state.observable_solvers[observable_tag]}
"""
    
    print("INSTRUCTIONS\n", instructions)

    ##############################

    updated_prop_code, obs_prop_code, _ = parameterModelCall(model, PropagatorConfig, "propagators", state.propagators, f"{observable_tag}_propagators", input_obs_prop_code, role, 
                                                             input_messages = [ HumanMessage(instructions) ], user_info_rules = user_info_rules, instance_validator=instanceCheck, group_validator=groupCheck)

    state.propagators = updated_prop_code

    if state.observable_propagators is None:
        state.observable_propagators = dict()
    state.observable_propagators[observable_tag] = obs_prop_code

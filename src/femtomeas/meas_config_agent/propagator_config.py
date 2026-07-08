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
from femtomeas.agent_common.python_output_agent import parameterModelCall

class PropagatorConfig(BaseModel):
    name : str = Field(..., description="The name/tag for the propagator instance")
    source: str = Field(..., description="The name/tag of the propagator's source instance")
    solver: str = Field(..., description="The name/tag of the propagator's solver instance")
    user_info: str = Field(..., description="Additional information (if any) provided by the user on what observables this propagator will be used for")
    
    def setXML(self,xml):
        opt = xml.addModule(self.name,"MFermion::GaugeProp")
        HadronsXML.setValues(opt, [ ("source",self.source), ("solver",self.solver) ])  

def identifyPropagators(model, state, user_interactions: list[BaseMessage]) -> str:
    role = """
You are responsible for identifying the (non-sink-smeared) lattice QCD propagators for the calculation and their associated solver and source. (Sink-smeared propagators are built from unsmeared propagators and will be created later.)

A propagator instance has a 'source' and 'solver' field that must be set, respectively, to the name of one of the source and solver instances identified previously.

First, identify the set of required propagators by first iterating over each of the previously identified observable instances noting how many propagators they require and other relevant information. Then, consider the message history to identify the source and solver combination that uniquely specifies each of the propagators needed for those observables. Also include propagators defined as inputs to sequential sources. Do not specify more propagators than are required for the observables.

If more than one observable requires a propagator with the same source/solver combination, you must re-use the propagator; do not create more propagators than needed.
    
- For each required propagator:
1. Identify the name of the associated source instance and use it to fill the 'source' parameter.
2. Identify the name of the associated solver instance and use it to fill the 'solver' parameter. To perform this identification, combine the parameters of the solver instance with those of the associated action instance, which is tagged by the field 'action'.
3. Assign a unique tag/name to the propagator instance. Never use the same tag for different instances. The tag should include the solver and source name.
4. For the 'user_info' field, summarize any information relevant to what observables this solver will be used for provided by the user. It is important that any positional information about the propagator be included, for example whether it is the first or second propagator of a two-point function, or if it is a 'spectator' quark in a baryon. If the user does now specify any details, use an empty string. For example, if the user specifies that this propagator will be used for both quarks of the pion two-point function, enter "use for both quarks of the pion two-point function" in user_info.

    
Propagator instance rules:    
- Your list must include every propagator instance required for the observables, and only those. Do not invent instances.
- You must reuse propagator instances that share the same source and solver
- Do not create separate non-sink-smeared propagators for different sink-smeared propagators. A single non-sink-smeared propagator can be sink smeared in arbitrary ways.
"""    

    def instanceCheck(prop):
        if not state.isValidSource(prop.source):
            return (False, f"\n-Source instance '{prop.source}' does not exist")        
        if not state.isValidSolver(prop.solver):
            return (False, f"\n-Solver instance '{prop.solver}' does not exist")
        return (True, "")

    def groupCheck(props):
        names = []
        valid = True
        invalid_why = ""
        for r in props:
            if r.name in names:
                invalid_why += f"\n-Propagator name '{r.name}' is not unique"
                valid = False
            names.append(r.name)
        return (valid, invalid_why)

    return parameterModelCall(model, PropagatorConfig, "propagators", role, input_messages = user_interactions, instance_validator=instanceCheck, group_validator=groupCheck)
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
from .common import *
from .hadrons_xml import HadronsXML

class PropagatorConfig(BaseModel):
    name : str = Field(..., description="The name/tag for the propagator instance")
    source: str = Field(..., description="The name/tag of the propagator's source instance")
    solver: str = Field(..., description="The name/tag of the propagator's solver instance")
    user_info: str = Field(..., description="Additional information (if any) provided by the user on what observables this propagator will be used for")

    #observables: List[str] = Field(...,description="The observable instances to which this propagator will be associated")
    
    def setXML(self,xml):
        opt = xml.addModule(self.name,"MFermion::GaugeProp")
        HadronsXML.setValues(opt, [ ("source",self.source), ("solver",self.solver) ])
    
class PropagatorsConfig(BaseModel):
    propagators: List[PropagatorConfig] = Field(...,description="The list of propagator instances")
   

def identifyPropagators(model, state, user_interactions: list[BaseMessage]) -> PropagatorsConfig:
    #Likely don't need an agent as we will not be asking questions of the user
    sys = """
You are responsible for identifying the lattice QCD propagators for the calculation alongside their associated solver and source.

A propagator instance has a 'source' and 'solver' field that must be set, respectively, to the name of one of the source and solver instances identified previously.

First, identify the set of required propagators by first iterating over each of the previously identified observable instances noting how many propagators they require and other relevant information. Then, consider the message history to identify the source and solver combination that uniquely specifies each of the propagators needed for those observables. Do not specify more propagators than are required for the observables.

If more than one observable requires a propagator with the same source/solver combination, you must re-use the propagator; do not create more propagators than needed.
    
- For each required propagator:
1. Identify the name of the associated source instance and use it to fill the 'source' parameter.
2. Identify the name of the associated solver instance and use it to fill the 'solver' parameter. To perform this identification, combine the parameters of the solver instance with those of the associated action instance, which is tagged by the field 'action'.
3. Assign a unique tag/name to the propagator instance. Never use the same tag for different instances. The tag should include the solver and source name.
4. For the 'user_info' field, summarize any information relevant to what observables this solver will be used for provided by the user. It is important that any positional information about the propagator be included, for example whether it is the first or second propagator of a two-point function, or if it is a 'spectator' quark in a baryon. If the user does now specify any details, use an empty string. For example, if the user specifies that this propagator will be used for both quarks of the pion two-point function, enter "use for both quarks of the pion two-point function" in user_info.

    
Propagator instance rules:    
- Your list must include every propagator instance required for the observables, and only those. Do not invent instances.
- You must reuse propagator instances that share the same source and solver
    
Your output must be in JSON format and adhere to the following schema:    
""" + json.dumps(PropagatorsConfig.model_json_schema())
  
    accepted = False
    obj = None
    while(accepted == False):
        obj = callModelWithStructuredOutput(model, sys, user_interactions, PropagatorsConfig, use_langchain_structured_output_method = False)

        #Auto validation
        valid = True
        invalid_why = "Your previous response was invalid for the following reason(s):"
        names = []
        for r in obj.propagators:
            if not state.isValidSource(r.source):
                invalid_why += f"\n-Source instance '{r.source}' does not exist"
                valid = False
            if not state.isValidSolver(r.solver):
                invalid_why += f"\n-Solver instance '{r.solver}' does not exist"
                valid = False
            if r.name in names:
                invalid_why += f"\n-Propagator name '{r.name}' is not unique"
                valid = False
            names.append(r.name)
                                
        if not valid:
            user_interactions.append(HumanMessage(invalid_why))
            continue


        #Human validation
        output = f"Obtained {len(obj.propagators)} propagator instances\n" + prettyPrintPydantic(obj.propagators)
        Print(output)

        accepted = queryYesNo("Is this correct?")
        if(accepted == False):
            reason = Input("Explain what is wrong: ")
            user_interactions.append(HumanMessage(f"Your previous response was not accepted for the following reason: {reason}"))            
    return obj

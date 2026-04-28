from langchain_core.messages import BaseMessage
from langchain_core.messages import (
    SystemMessage,
    HumanMessage,
    ToolCall,
    AIMessage
)

from pydantic import BaseModel, Field, ConfigDict, NonNegativeInt, TypeAdapter
from typing import Literal, Union, List, Optional, Tuple
from langchain.agents.structured_output import ToolStrategy, ProviderStrategy
from .common import *
from .hadrons_xml import HadronsXML

def mesonModuleXML(name, xml, gammas_snk_src: str, q1 : str, q2 : str):
    #NB: gamma5-hermiticity used on q2
    opt = xml.addModule(name, "MContraction::Meson")

    HadronsXML.setValues(opt, [ ("q1", q1), ("q2", q2), ("gammas", gammas_snk_src), ("sink", "point_sink_zerop"), ("output",f"{name}.out") ])

def baryonModuleXML(name, xml, gammas_snk_src: str, q1 : str, q2 : str, q3 : str):
    opt = xml.addModule(name, "MContraction::Baryon")

    HadronsXML.setValues(opt, [ ("q1", q1), ("q2", q2), ("q3", q3), ("gammas", gammas_snk_src), ("sink", "point_sink_zerop"), ("output",f"{name}.out") ])

def validateProps(prop_names, state):
    for p in prop_names:
        if not state.isValidPropagator(p):
            return (False, f"-Propagator instance {p} does not exist")
    return (True,"")

    
class Pion2ptConfig(BaseModel):
    """An instance of the pion two-point function calculation."""
    type: Literal["pion2pt"] = "pion2pt"
    propagators : tuple[str,str] = Field(..., description="The tags of the propagators used to compute the observable")

    def setXML(self, name, xml):
        mesonModuleXML(name, xml, "(Gamma5 Gamma5)", self.propagators[0], self.propagators[1])
       
    def validate(self, state):
        return validateProps(self.propagators,state)
       
   
class Vector2ptConfig(BaseModel):
    """An instance of the vector two-point function calculation."""
    type: Literal["vector2pt"] = "vector2pt"
    propagators : tuple[str,str] = Field(..., description="The tags of the propagators used to compute the observable")

    def setXML(self, name, xml):
        gammas = ""
        for gsnk in ("GammaX","GammaY","GammaZ"):
            for gsrc in ("GammaX","GammaY","GammaZ"):
                gammas = gammas + f"({gsnk} {gsrc})"
       
        mesonModuleXML(name, xml, gammas, self.propagators[0], self.propagators[1])

    def validate(self, state):
        return validateProps(self.propagators, state)

class Nucleon2ptConfig(BaseModel):
    """An instance of the nucleon two-point function calculation."""
    type: Literal["nucleon2pt"] = "nucleon2pt"
    propagators : tuple[str,str,str] = Field(..., description="The tags of the three propagators used to compute the observable")

    def setXML(self, name, xml):
        baryonModuleXML(name, xml, "(CG5 CG5)", self.propagators[0], self.propagators[1], self.propagators[2])

    def validate(self, state):
        return validateProps(self.propagators, state)

class ObservableConfig(BaseModel):
    """An instance of an observable."""
    name: str = Field(..., description="The name/tag of the observable instance")        
    obs: Union[Pion2ptConfig,Vector2ptConfig,Nucleon2ptConfig] = Field(...,description="The observation instance and configuration.", discriminator='type')

    def setXML(self, xml):
        self.obs.setXML(self.name, xml)

    def validate(self, state):
        obs_info = state.locateObservable(self.name)
        if obs_info is None:
            return (False,"The name/tag for this observable instance does not match one in the list of previously-identified observables")
        if obs_info.obs_type.type != self.obs.type:
            return (False,"The 'obs' field for this observable instance does not have the same type as the ObservableInfo with this name")
        
        return self.obs.validate(state)
       
class ObservablesConfig(BaseModel):
    observable_configs: List[ObservableConfig] = Field(...,description="The list of observable instances and their configurations")


def configureObservables(model, state, user_interactions: list[BaseMessage]) -> ObservablesConfig:
    sys = """
    Reasoning: high

    You are an assistant responsible for building a list of lattice QCD observable instances and their associated parameters based on the conversation history.

    In previous stages of the workflow, agents identified a list of observables that will be computed alongside some associated information. For each and every observable in this list you must determine the associated propagators and other parameters.

    Your workflow:

    For every ObservableInfo in the list contained within the message history:
    1. Parse the user information and background knowledge for the observable
    2. Create a new ObservableConfig and use the 'name' field from the ObservableInfo to fill in the 'name' field for this observable instance. Do not generate new observable names.
       Select the type of the 'obs' field based on the 'obs_type' field of the ObservableInfo, ensuring that their 'type' fields match.
    3. Identify the propagators required to compute this observable and note their 'name' field based upon the 'user_info' field of the PropagatorConfig. These names and no other must be used to fill the list of propagators in the 'obs' field. Use only the names of propagators, not of other types of instance (e.g. sources, solvers, actions)
    
    Your list must include every observable in the list and only those. Do not invent observables, do not combine observables, and do not add details that are not explicitly provided by the user.
    Do not invent or infer any information not explicitly obtained from the message history.
    Do not invent names for propagators, use only those assigned to existing propagators in your message history.
"""

    accepted = False
    obj = None
    while(accepted == False):
        obj = callModelWithStructuredOutput(model, sys, user_interactions, ObservablesConfig, True)

        #Auto validation
        valid = True
        invalid_why = "Your previous response was invalid for the following reason(s):"
        names = []
        for r in obj.observable_configs:
            val, rsn = r.validate(state)
            if not val:
                invalid_why += "\n" + rsn
                valid = False
            if r.name in names:
                invalid_why += f"\n-Propagator name '{r.name}' is not unique"
                valid = False
            names.append(r.name)
                                
        if not valid:
            user_interactions.append(HumanMessage(invalid_why))
            continue

        
        #Human validation
        output = f"Obtained {len(obj.observable_configs)} observable configuration instances\n" + prettyPrintPydantic(obj.observable_configs)
        Print(output)
            
        accepted = queryYesNo("Is this correct?")
        if(accepted == False):
            reason = Input("Explain what is wrong: ")
            user_interactions.append(HumanMessage(f"Your previous response was not accepted for the following reason: {reason}"))
    return obj

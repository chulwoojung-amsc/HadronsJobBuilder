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
from femtomeas.agent_common.common import *
from .hadrons_xml import HadronsXML
from femtomeas.agent_common.agent_base import parameterAgent

Gammas = Literal["MinusGamma5", "Gamma5", "MinusGammaT", "GammaT", "MinusGammaTGamma5", "GammaTGamma5", "MinusGammaX", "GammaX", "MinusGammaXGamma5", "GammaXGamma5", "MinusGammaY", "GammaY", "MinusGammaYGamma5", "GammaYGamma5", "MinusGammaZ", "GammaZ", "MinusGammaZGamma5", "GammaZGamma5", "MinusIdentity", "Identity", "MinusSigmaXT", "SigmaXT", "MinusSigmaXY", "SigmaXY", "MinusSigmaXZ", "SigmaXZ", "MinusSigmaYT", "SigmaYT", "MinusSigmaYZ", "SigmaYZ", "MinusSigmaZT", "SigmaZT"]

mesonSpecialKeywords = Literal["pion","kaon","pseudoscalar","vector","axial-vector"]

def getMesonGammas(op : mesonSpecialKeywords)-> List[Gammas]:
    if op in ("pion","kaon","pseudoscalar"):
        return ["Gamma5"]
    elif op == "vector":
        return ["GammaX","GammaY","GammaZ"]
    elif op == "axial-vector":
        return ["GammaXGamma5","GammaYGamma5","GammaZGamma5"]
    else:
        raise Exception("Unknown meson operator")

@tool
def getMesonGammasTool(op : mesonSpecialKeywords)-> List[Gammas]:
    """Get the list of Gammas (combination of Gamma-matrices in the Euclidean Clifford algebra) associated with a particular meson operator in the list of special meson keywords"""
    return getMesonGammas(op)


def validateProps(prop_names, state):
    for p in prop_names:
        if not state.isValidPropagator(p):
            return (False, f"-Propagator instance {p} does not exist")
    return (True,"")


class Meson2ptConfig(BaseModel):
    """A meson two-point function or "correlator"."""    
    type: Literal["meson2pt_config"] = "meson2pt_config"
    
    propagators : Tuple[str,str] = Field(..., description="The tags of the propagators used to compute the observable")
    sink_gammas: List[Gammas] = Field(...,description="The list of Gamma-matrix combinations to use at the sink")
    source_gammas: List[Gammas] = Field(...,description="The list of Gamma-matrix combinations to use at the sink")
    
    def setXML(self, name, xml):
        gammas_snk_src = ""
        for gsnk in self.sink_gammas:
            for gsrc in self.source_gammas:
                gammas_snk_src = gammas_snk_src + f"({gsnk} {gsrc})"
        
        #NB: gamma5-hermiticity used on q2
        opt = xml.addModule(name, "MContraction::Meson")

        HadronsXML.setValues(opt, [ ("q1", self.propagators[0]), ("q2", self.propagators[1]), ("gammas", gammas_snk_src), ("sink", "point_sink_zerop"), ("output",f"{name}.out") ])
       
    def check(self, state):
        return validateProps(self.propagators,state)
    
   
class ObservableConfig(BaseModel):
    """An instance of an observable."""
    name: str = Field(..., description="The name/tag of the observable instance")        
    obs: Union[Meson2ptConfig] = Field(...,description="The observation instance and configuration.", discriminator='type')

    def setXML(self, xml):
        self.obs.setXML(self.name, xml)

    def check(self, state):
        return self.obs.check(state)
       
class ObservablesConfig(BaseModel):
    observable_configs: List[ObservableConfig] = Field(...,description="The list of observable instances and their configurations")


def configureObservables(model, state, user_interactions: list[BaseMessage]) -> ObservablesConfig:
    role = """for building a list of lattice QCD observable instances and their associated parameters based on the conversation history.

    In previous stages of the workflow, agents identified a list of observable types that will be computed alongside some associated information. For each and every observable type in this list you must instantiate the required number of observable instances and determine their propagators and other parameters."""

    parameter_rules = ["""observable_configs:

  Perform the following workflow:

  For every ObservableInfo in the list contained within the message history:
    1. Parse the user information and background knowledge for the observable
    2. Determine how many ObservableConfig instances are required to compute the observable for all different parameter combinations specified by the user. Follow the rules below.
    3. Instantiate the ObservableConfig instances, populate their parameters and add them to 'observable_configs',

  Rules for ObservableConfig instances:
  - A separate ObservableConfig is required for each unique combination of parameters, even if the observable itself is the same. For example, if the user wants to compute the pion 2pt function with two different combinations of propagators, create two separate instances of ObservableConfig.
  - Your list must include every observable in the list and only those. Do not invent observables, do not combine observables, and do not add details that are not explicitly provided by the user.
  - Do not invent or infer any information not explicitly obtained from the message history.
  - Do not invent names for propagators, use only those assigned to existing propagators in your message history.""",

    """ObservableConfig.name:
  - You must assign a unique tag/name to the instance. Do not ask the user for this parameter
  - Never use the same tag for different instances.
  - The tag should include the observable type and enough of the parameter values to uniquely distinguish it among the other ObservableConfig instances, prefering shorter tags if possible.""",
                       
    """Meson2ptConfig.sink_gammas and Meson2ptConfig.source_gammas :
    - These are lists of Gamma-matrix combinations for the sink and source locations, respectively.
    - If the user has not previously provided these parameters, perform the following workflow:
      1) Based upon the observable type and other user-provided information, attempt to identify the special "mesonSpecialKeywords" keywords that describe the meson states at the source and sink. You can use the same keyword for the source and sink mesons unless the user has specified otherwise.

      2) - If you are able to identify the keywords, you must describe the keywords you identified for both source and sink to the user, and ask them to confirm. Ensure you specify both source and sink keywords even if they are the same. Do not ask more than one question at a time. You can ask the user to confirm multiple keywords at once, but only in the form of a single question.
         - If you are *not* able to identify the keywords, ask the user to either choose the keywords or else manually specify the lists of Gamma-matrix combinations at the source and sink

      3) - If you have identified or have been given the keywords, you must call the getMesonGammasTool tool to obtain the list of Gamma-matrix combinations for the source and sink. 
         - Otherwise, if the user specified the Gamma-matrix combinations, use those to populate sink_gammas and source_gammas and finish this workflow
         - Never guess the gamma matrix combinations""",

    """*.propagators (where * applies to any observable type that involves propagators:
  - Use the message history and the skills descriptions of the observables to identify the propagators required to compute this observable and note their 'name' fields. Use only the names of propagators, not of other types of instance (e.g. sources, solvers, actions)"""
                       ]

    tools = [getMesonGammasTool]
    tool_rules = []
    
    return parameterAgent(model, ObservablesConfig, role, tools=tools, tool_rules=tool_rules, parameter_rules=parameter_rules, input_messages=user_interactions)

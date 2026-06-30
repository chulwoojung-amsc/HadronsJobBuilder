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
from femtomeas.meas_config_agent.hadrons_xml import HadronsXML
from femtomeas.agent_common.python_output_agent import parameterAgent
from femtomeas.meas_config_agent.meas_agent_common import Gammas
from femtomeas.meas_config_agent.source_config import momentumStr

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

#TODO: cache and reuse these? Don't think there's much need but it makes neater XML
class ContractionSinkPoint(BaseModel):
    """A point sink with optional momentum (default zero-momentum)"""
    type: Literal["contraction_sink_point"] = "contraction_sink_point"
    momentum : Tuple[float,float,float,float] | None = Field(..., description="An optional four-momentum")

    def setXML(self,obs_name,xml):
        name = obs_name + "_sinkpoint"
        snk = xml.addModule(name, "MSink::ScalarPoint")
        HadronsXML.setValue(snk, "mom", momentumStr(self.momentum))
        return name

    def check(self, state):        
        return (True, "")
      
class ContractionSinkNone(BaseModel):
    """Use this sink type when using smeared propagators"""
    type: Literal["contraction_sink_none"] = "contraction_sink_none"

    def setXML(self,obs_name,xml):
        return ""

    def check(self, state):        
        return (True, "")

class Meson2ptInstance(BaseModel):
    """An instance of a calculation of a meson two-point function with a specific propagator and sink combination"""
    sink : Union[ContractionSinkNone, ContractionSinkPoint] = Field(..., description="The sink smearing for contracting unsmeared propagators", discriminator='type')
    propagators : Tuple[str,str] = Field(..., description="The tags of the propagators used to compute the observable")
    name: str = Field(..., description="The name/tag of the observable instance")        

    def setXML(self, gammas_snk_src, xml):
        sink_nm = self.sink.setXML(self.name, xml)

        #NB: gamma5-hermiticity used on q2
        opt = xml.addModule(self.name, "MContraction::Meson")
        HadronsXML.setValues(opt, [ ("q1", self.propagators[0]), ("q2", self.propagators[1]), ("gammas", gammas_snk_src), ("sink", sink_nm), ("output",f"{self.name}.out") ])

    def check(self, state):
        lprop_exists = state.isValidPropagator(self.propagators[0])
        rprop_exists = state.isValidPropagator(self.propagators[1])
        lsprop_exists = state.isValidSmearedPropagator(self.propagators[0])
        rsprop_exists = state.isValidSmearedPropagator(self.propagators[1])

        if not lprop_exists and not lsprop_exists:
            return (False, f"Propagator {self.propagators[0]} does not exist")
        if not rprop_exists and not rsprop_exists:
            return (False, f"Propagator {self.propagators[1]} does not exist")
        
        if (lprop_exists and not rprop_exists) or (rprop_exists and not lprop_exists) or (lsprop_exists and not rsprop_exists) or (rsprop_exists and not lsprop_exists):
            return (False, "Cannot mix smeared and unsmeared propagators in meson contractions")

        if (lsprop_exists and not isinstance(self.sink, ContractionSinkNone) ):
            return (False, "Sink argument must be ContractionSinkNone when using smeared propagators")
        if (lprop_exists and isinstance(self.sink,ContractionSinkNone) ):
            return (False, "Sink argument must be different from ContractionSinkNone when using unsmeared propagators")
        
        return (True, "")

class Meson2ptConfig(BaseModel):
    """A meson two-point function or "correlator"."""    
    type: Literal["meson2pt_config"] = "meson2pt_config"

    instances: List[Meson2ptInstance] = Field(..., description="Instances of this observable with different combinations of propagators")
    sink_gammas: List[Gammas] = Field(...,description="The list of Gamma-matrix combinations to use at the sink")
    source_gammas: List[Gammas] = Field(...,description="The list of Gamma-matrix combinations to use at the sink")
    
    def setXML(self, xml):
        gammas_snk_src = ""
        for gsnk in self.sink_gammas:
            for gsrc in self.source_gammas:
                gammas_snk_src = gammas_snk_src + f"({gsnk} {gsrc})"

        for instance in self.instances:                
            instance.setXML(gammas_snk_src, xml)
       
    def check(self, state):
        result = True
        reason = ""

        if len(self.sink_gammas) == 0 or len(self.source_gammas) == 0:
            result = False
            reason += "\nBoth source and sink must have at least one Gamma-matrix combination"
        
        for instance in self.instances:        
            r = instance.check(state)
            if not r[0]:
                result = False
                reason = reason + "\n" + r[1]            

        return (result, reason)
   

class WritePropagators(BaseModel):
    """List of propagators to write to disk and their filestems (filename without .${CFG}.bin extension)"""
    type: Literal["write_propagators"] = "write_propagators"
    write_props : List[ Tuple[str,str] ] = Field(..., description="List of propagator name, local filestem pairs")

    def setXML(self, xml):
        for p in self.write_props:
            nm = p[0] + "_write"
            opt = xml.addModule(nm, "MIO::SavePropagator")
            HadronsXML.setValues(opt, [ ("name", p[0]), ("fileStem", p[1]) ])

    def check(self, state):
        props = [p[0] for p in self.write_props]
        return validateProps(props)


class ObservableConfig(BaseModel):
    """An instance of an observable."""
    obs: Union[Meson2ptConfig, WritePropagators] = Field(...,description="The observation instance and configuration.", discriminator='type')

    def setXML(self, xml):
        self.obs.setXML(xml)

    def check(self, state):
        return self.obs.check(state)     

def configureObservables(model, state, user_interactions: list[BaseMessage]) -> ObservableConfig:
    role = """for building a list of lattice QCD observable instances and their associated parameters based on the conversation history.

    In previous stages of the workflow, agents identified a list of observable types that will be computed alongside some associated information. For each and every observable type in this list you must instantiate the required number of observable instances and determine their propagators and other parameters."""

    parameter_rules = ["""observable_configs:

  Perform the following workflow:

  For every ObservableInfo in the list contained within the message history:
    1. Parse the user information and background knowledge for the observable
    2. Determine the ObservableConfig instances required to compute all observable types specified by the user. Follow the rules below.
    3. Instantiate the ObservableConfig instances, populate their parameters and add them to 'observable_configs',

  Rules for ObservableConfig instances:
  - A separate ObservableConfig is required for each unique combination of parameters other than propagators (these are treated separately using the "instances" parameter), even if the observable class is the same. For example, if the user wants to compute the pion and vector two-point functions, create two instances of ObservableConfig, one for the pion and one for the vector. 
  - Your list must include every observable in the list and only those. Do not invent observables, do not combine observables, and do not add details that are not explicitly provided by the user.
  - Do not invent or infer any information not explicitly obtained from the message history.""",

    """Meson2ptConfig.instances:
  - Create a different instance for each unique combination of propagators
  """,

    """Meson2ptInstance.name:
  - You must assign a unique tag/name to the instance. Do not ask the user for this parameter
  - Never use the same tag for different instances.
  - The tag should include the observable type and enough of the parameter values to uniquely distinguish it among the other instances, prefering shorter tags if possible.""",
                       
    """Meson2ptInstance.propagators:
  - Use the message history and the skills descriptions of the observables to identify the propagators required to compute this observable and note their 'name' fields. Use only the names of propagators, not of other types of instance (e.g. sources, solvers, actions)
  - Do not invent names for propagators, use only those assigned to existing propagators in your message history.
  - You MUST ensure the order of the two propagators in the Tuple matches the role of the two quarks. For example, if the user says "prop_1" should be used as the first (or incoming) quark, ensure it is the first entry.
  - You can use either two smeared propagators or two unsmeared propagators, but you cannot mix smeared and unsmeared propagators
  """,

    """Meson2ptInstance.sink:
  - When using smeared propagators, you must use a ContractionSinkNone for this parameter
  - When using regular/unsmeared propagators, you must choose the sink type according to the user's input. The most common sink type is ContractionSinkPoint (point sink); you may suggest this to the user.""",
                       
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
    
    def instanceCheck(obs):
        return obs.check(state)

    return parameterAgent(model, ObservableConfig, role, tools=tools, tool_rules=tool_rules, parameter_rules=parameter_rules, input_messages=user_interactions, instance_validator=instanceCheck)

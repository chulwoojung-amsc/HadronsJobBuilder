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
from langchain.agents import create_agent
from femtomeas.meas_config_agent.hadrons_xml import HadronsXML
import json
from femtomeas.agent_common.common import *
from femtomeas.agent_common.python_output_agent import parameterAgent
from femtomeas.meas_config_agent.meas_agent_common import Gammas
from .source_config import momentumStr

class WallSmear(BaseModel):
    """A wall smearing with optional momentum,    sum_x e^{+i p . x} prop_sol(x)  """
    type: Literal["wall_smear"] = "wall_smear"
    momentum : Tuple[float,float,float,float] | None = Field(..., description="An optional four-momentum")

    def setXML(self,smeared_prop_name,prop_name,xml):
        smr_base_name = smeared_prop_name + "_base"
        base = xml.addModule(smr_base_name, "MSink::Point")
        HadronsXML.setValue(base, "mom", momentumStr(self.momentum))
        
        opt = xml.addModule(smeared_prop_name,"MSink::Smear")
        HadronsXML.setValues(opt, [("sink", smr_base_name), ("q", prop_name)] )

    def check(self, state):
        return (True, "")


class SmearedPropagatorConfig(BaseModel):
    name : str = Field(..., description="The name/tag for the smeared propagator")
    input_prop : str = Field(..., description="The name/tag of the input propagator")
    smearing: Union[WallSmear] = Field(
        ..., description="Information about the smearing type", discriminator='type'
    )
    user_info: str = Field(..., description="Additional information (if any) provided by the user on what observables this smeared propagator will be used for")
    
    def setXML(self,xml):
        self.smearing.setXML(self.name, self.input_prop,  xml)

    def check(self, state):
        if not state.isValidPropagator(self.input_prop):
            return (False, f"Input propagator {self.input_prop} does not exist")
        return self.smearing.check(state)

def identifySmearedPropagators(model, state, user_interactions: list[BaseMessage]) -> str:
    role = """creating configurations for each smeared propagator required by the user.

    Previous agent interactions have identified a set of observables and their required propagators. Propagators may be smeared at their sink location prior to being contracted. You must create SmearedPropagatorConfig instances for each required smeared propagator.     

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
  - The tag should include the smearing type and enough of the parameter values to uniquely distinguish it among the other instances, prefering shorter tags if possible.""",
        """SmearedPropagator.user_info:
  - For the 'user_info' field, you must summarize any information relevant to what observables this smeared propagator will be used for provided by the user. Never ask the user to provide this parameter.
  - For observables with more than one propagator argument, you must record for which propagator the smeared propagator is to be used.
  - If the user does now specify any details, use an empty string.""" ]
            
    
    additional_user_query_rules = [ ]
    
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

    return parameterAgent(model, SmearedPropagatorConfig, "smeared_propagators", role, tools=[], input_messages=user_interactions, parameter_rules=parameter_rules, group_validator=checkAll )
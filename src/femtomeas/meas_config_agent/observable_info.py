from langchain_core.messages import BaseMessage
from langchain.messages import (
    SystemMessage,
    HumanMessage,
    ToolCall,
    AIMessage
)
from langchain.agents import create_agent
from pydantic import BaseModel, Field, ConfigDict, NonNegativeInt, TypeAdapter
from typing import Literal, Union, List, Optional, Tuple
from femtomeas.agent_common.common import *
from femtomeas.agent_common.agent_base import parameterAgent

class Meson2ptObs(BaseModel):
   """A meson two-point function or "correlator"."""
   type: Literal["meson2pt"] = "meson2pt"
   meson_type : str = Field(..., description="The type of meson, e.g. pion, kaon, rho. Also accept types described by their parity transformation properties, e.g. scalar, pseudoscalar, vector")

   def skill(self):
      return """- Meson two-point functions:
  Meson two-point functions are typically used to compute particle masses or decay constants (e.g. f_pi).

  This observable requires two propagators, that are contracted together at some timeslice-localized sink. The first propagator argument has quark flow from source to sink, and the second propagator argument has gamma^5-hermiticity applied to it such that the quark flow is from sink to source. We often refer to these propagators by the direction of quark flow into the vertex, i.e. "incoming" for the first quark and "outgoing" for the second.

  Pion and kaon correlators are both pseudoscalar two-point functions, the difference being that the pion usually has both quarks with the same (light) mass, whereas the kaon has one heavy and one light quark."""
   
class ObservableInfo(BaseModel):
   """Information about an observable to be computed."""
   obs_type: Union[Meson2ptObs]= Field(...,description="The observable contraction and particle types, and important knowledge.", discriminator="type")


def observableSkills(observables):
   inc = set()
   out = """
--------------------------------------
Information about required observables
--------------------------------------
"""
   for o in observables:
      if (t := type(o.obs_type)) not in inc:
         out = out + o.obs_type.skill() + "\n"
         inc.add(t)
   print("SKILLS", out)
   return out


   
class ObservablesInfo(BaseModel):
    observables: List[ObservableInfo] = Field(...,description="The list of observables")

    def check(self):
       return (True, "")

    
def identifyObservables(model, user_interactions: list[BaseMessage]) -> ObservablesInfo:
   role = """identifying all lattice QCD observables the user wants to compute

   You will receive the user’s description of what observables they wish to compute, and your task is to use this information to fill the output data structure.
   Rather than specifying observables, the user may ask you questions; respond to those questions as appropriate then ask the user again to describe the observables they want to compute. Use this information to populate your output."""


   parameter_rules = [
      """observables:

  You must analyze the user's input description to identify which observables the user wants to compute and add an ObservableInfo instance for each distinct observable to "observables".
      
  Rules:
  - Any given observable can appear only once, even if the user wants to compute it multiple times with different inputs.
  - Your list must include every observable explicitly mentioned, and only those observables. Do not invent observables, do not combine observables unless the user explicitly describes them as the same, and do not add details that are not explicitly provided by the user.
  - Do not ask the user if they want to specify any more observables.
  - Do not ask the user to confirm the list of observables
  
  """,
  
  "obs_type: Populate the 'obs_type' field with an object of type appropriate to the observable. If the user describes an observable that is not supported, you must describe to the user which observables you support and ask the user which ones they want."

    ]

   return parameterAgent(model, ObservablesInfo, role, tools=[], tool_rules=[], parameter_rules=parameter_rules, input_messages=user_interactions)

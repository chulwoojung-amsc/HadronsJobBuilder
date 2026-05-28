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
   user_info: str = Field(...,description="Any relevant information obtained from the user regarding the observable, such as "
                     "propagator masses, momenta, source/sink smearing, etc. "
                     "Use an empty string if no extra information is given.")


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
   """
   Parse the list of messages to identify a list of observable keys and their associated information
   """

   role = "identifying all lattice QCD observables the user wants to compute, and extracting only the information explicitly provided by the user that is relevant to computing each observable."


   parameter_rules = [
      """observables:
  You will receive the user’s original request. Your task is to read only this content and produce a structured list of observables in the 'observables' field of your output. Do not invent, infer, or assume any information that is not explicitly stated by the user.

  Use the following workflow:
  - From the user's response, identify which observables the user wants to compute.
  - Record any other information provided by the user about that observable in the user_info field
      
  Rules:
  - Any given observable can appear only once, even if the user wants to compute it multiple times with different inputs.
  - Your list must include every observable explicitly mentioned, and only those observables. Do not invent observables, do not combine observables unless the user explicitly describes them as the same, and do not add details that are not explicitly provided by the user.
  - Do not ask the user if they want to specify any more observables.
  - Do not ask the user to confirm the list of observables       
  """,

  """user_info: In the 'user_info' field, you must summarize any additional information provided by the user regarding the observable. Record only the information that the user has clearly provided about that specific instance of the observable. NEVER ask the user for this value. NEVER ask the user to provide additional information. 

  Examples include:
  – required propagators
  – operator insertions
  – quantum numbers or kinematic parameters
  – anything else explicitly tied to the computation
  If the user did not specify extra information for an observable, leave the user_info field empty rather than guessing or filling in defaults. Never ask the user to provide additional information.""",
  
  "obs_type: Populate the 'obs_type' field with an object of type appropriate to the observable. If the user describes an observable that is not supported, you must describe to the user which observables you support and ask the user which ones they want."

    ]

   return parameterAgent(model, ObservablesInfo, role, tools=[], tool_rules=[], parameter_rules=parameter_rules, input_messages=user_interactions)

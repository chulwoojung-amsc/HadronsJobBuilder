from langchain_core.messages import BaseMessage
from langchain.messages import (
    SystemMessage,
    HumanMessage,
    ToolCall,
    AIMessage
)

from pydantic import BaseModel, Field, ConfigDict, NonNegativeInt, TypeAdapter
from typing import Literal, Union, List, Optional, Tuple
from .common import *

class Pion2ptObs(BaseModel):
   """The pion two-point function. This observable involves a contraction of two propagators, which may be the same."""
   type: Literal["pion2pt"] = "pion2pt"
   n_propagator: Literal[2] = Field(2, description="The required number of propagators")
   obs_info: Literal[""] = Field("", description="General information about this observable")   
   
class Vector2ptObs(BaseModel):
   """The vector two-point function. This observable involves a contraction of two propagators, which may be the same."""
   type: Literal["vector2pt"] = "vector2pt"
   n_propagator: Literal[2] = Field(2, description="The required number of propagators")
   obs_info: Literal[""] = Field("", description="General information about this observable")
   
class ObservableInfo(BaseModel):
   """Information about an observable to be computed."""
   obs_type: Union[Pion2ptObs,Vector2ptObs] = Field(...,description="The observation type and important knowledge.", discriminator="type")
   user_info: str = Field(...,description="Any relevant information obtained from the user regarding the observable, such as "
                     "propagator masses, momenta, source/sink smearing, etc. "
                     "Use an empty string if no extra information is given.")
   name: str = Field(...,description="A unique name/tag identifier for this observable instance")
    
class ObservablesInfo(BaseModel):
    observables: List[ObservableInfo] = Field(...,description="The list of observables")

def identifyObservables(model, user_interactions: list[BaseMessage]) -> ObservablesInfo:
   """
   Parse the list of messages to identify a list of observable keys and their associated information
   """
    
   sys = """
   Reasoning: high
    
   You are an assistant responsible for identifying all lattice QCD observables the user wants to compute, and extracting only the information explicitly provided by the user that is relevant to computing each observable.

You will receive the user’s original request. Your task is to read only this content and produce a structured list of observables. Do not invent, infer, or assume any information that is not explicitly stated by the user.

For each observable mentioned by the user:

Create a separate entry for each, even if the observable appears multiple times with different parameters or conditions.

Assign a unique name/tag to the observable instance.
    
In the user_info field, summarize any additional information provided by the user regarding the observable. Record only the information that the user has clearly provided about that specific instance of the observable. Examples include:
– required propagators
– operator insertions
– quantum numbers or kinematic parameters
– anything else explicitly tied to the computation   
If the user did not specify information for an observable, leave its user_info field empty rather than guessing or filling in defaults.

You can output the same observable information for multiple entries but only if they have different observable types.
    
Your list must include every observable explicitly mentioned, and only those observables. Do not invent observables, do not combine observables unless the user explicitly describes them as the same, and do not add details that are not explicitly provided by the user.
"""

   accepted = False
   obj = None
   while(accepted == False):
      obj = callModelWithStructuredOutput(model, sys, user_interactions, ObservablesInfo, True)

      output = f"Obtained {len(obj.observables)} observables:\n" + prettyPrintPydantic(obj.observables)
      Print(output)
            
      accepted = queryYesNo("Is this correct?")
      if(accepted == False):
         reason = Input("Explain what is wrong: ")
         user_interactions.append(HumanMessage(f"Your previous response was not accepted for the following reason: {reason}"))            
   return obj
 

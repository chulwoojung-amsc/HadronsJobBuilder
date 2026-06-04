from langchain_core.messages import BaseMessage
from langchain_core.messages import (
    SystemMessage,
    HumanMessage,
    ToolCall,
    AIMessage
)
from langchain.agents import create_agent
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

class Nucleon2ptObs(BaseModel):
   """The nucleon two-point function. This observable involves a contraction of three propagators, which may be the same."""
   type: Literal["nucleon2pt"] = "nucleon2pt"
   n_propagator: Literal[3] = Field(3, description="The required number of propagators")
   obs_info: Literal[""] = Field("", description="General information about this observable")

# AUTO-GENERATED BEGIN
class Delta2ptXObs(BaseModel):
   """Delta baryon 2pt function (spin-3/2, x polarisation). j32X interpolator."""
   type: Literal["delta2pt_x"] = "delta2pt_x"
   n_propagator: Literal[3] = Field(3, description="The required number of propagators")
   obs_info: Literal[""] = Field("", description="General information about this observable")

class Delta2ptYObs(BaseModel):
   """Delta baryon 2pt function (spin-3/2, y polarisation). j32Y interpolator."""
   type: Literal["delta2pt_y"] = "delta2pt_y"
   n_propagator: Literal[3] = Field(3, description="The required number of propagators")
   obs_info: Literal[""] = Field("", description="General information about this observable")

class Delta2ptZObs(BaseModel):
   """Delta baryon 2pt function (spin-3/2, z polarisation). j32Z interpolator."""
   type: Literal["delta2pt_z"] = "delta2pt_z"
   n_propagator: Literal[3] = Field(3, description="The required number of propagators")
   obs_info: Literal[""] = Field("", description="General information about this observable")

# AUTO-GENERATED END

class ObservableInfo(BaseModel):
   """Information about an observable to be computed."""
   obs_type: Union[Pion2ptObs, Vector2ptObs, Nucleon2ptObs, Scalar2ptObs, Axialvector2ptObs, Temporalvector2ptObs, Axialtemporalvector2ptObs, Tensor2ptObs, Delta2ptObs, Delta2ptXObs, Delta2ptYObs, Delta2ptZObs] = Field(...,description="The observation type and important knowledge.", discriminator="type")
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

You will receive the user’s original request. Your task is to read only this content and produce a structured list of observables in the 'observables' field of your output. Do not invent, infer, or assume any information that is not explicitly stated by the user.
  
For each observable mentioned by the user:
1) Create a separate ObservableInfo entry for each, even if the observable appears multiple times with different parameters or conditions.
2) Assign a unique name to the observable instance in the 'name' field
3) In the 'user_info' field, summarize any additional information provided by the user regarding the observable. Record only the information that the user has clearly provided about that specific instance of the observable. Examples include:
     – required propagators
     – operator insertions
     – quantum numbers or kinematic parameters
     – anything else explicitly tied to the computation
  If the user did not specify information for an observable, leave its user_info field empty rather than guessing or filling in defaults.
4) Populate the 'obs_type' field with an object of type appropriate to the observable. If the user describes an observable that is not supported, you must describe to the user which observables you support and ask the user which ones they want. Use the User Query rules below to inform/ask the user.

Observable instance rules:
- You can output the same observable information for multiple entries but only if they have different observable types.
- Your list must include every observable explicitly mentioned, and only those observables. Do not invent observables, do not combine observables unless the user explicitly describes them as the same, and do not add details that are not explicitly provided by the user.

User Query rules:
- Use the getUserInput tool to ask questions of the user
- If the user responds to a query with an invalid response, repeat the query until a valid response is provided. Never accept an invalid response.
- Instead of answering your question, the user might respond to your query with a question. If this occurs, answer the user's question using provideInformationToUser tool and ensure the user is satisfied with a follow-up call to getUserInput. Once satisfied, repeat the original question.

Your output must be in JSON format and adhere to the following schema:    
""" + json.dumps(ObservablesInfo.model_json_schema())

   accepted = False
   obj = None
   while(accepted == False):
      agent = create_agent(model=model, tools=[getUserInput,provideInformationToUser], system_prompt=sys, response_format=ObservablesInfo)
        
      resp = agent.invoke({ "messages": user_interactions },     {"configurable": {"thread_id": "1"}})
      obj = resp["structured_response"]
      user_interactions = resp['messages']

      output = f"Obtained {len(obj.observables)} observables:\n" + prettyPrintPydantic(obj.observables)
      Print(output)
            
      accepted = queryYesNo("Is this correct?")
      if(accepted == False):
         reason = Input("Explain what is wrong: ")
         user_interactions.append(HumanMessage(f"Your previous response was not accepted for the following reason: {reason}"))            
   return obj
 

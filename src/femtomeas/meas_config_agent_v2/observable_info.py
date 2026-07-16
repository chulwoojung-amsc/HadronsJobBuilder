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
from .observable_info_models import ObservablesInfo
from .state import State

def identifyObservables(model, user_input: str, state: State)->ObservablesInfo:
   current_obs = ObservablesInfo(observables=[]) if state.observables is None else state.observables

   role = f"""identifying all lattice QCD observables the user wants to compute

You will receive the user’s description of what observables they wish to compute, and your task is to use this information to identify any new entries required in ObservablesInfo.observables.

The known observables are as follows:
---------------------------------------------
{current_obs.model_dump_json()}
---------------------------------------------
For each observable specified by the user, check to see if an existing ObservableInfo entry matches. If no existing entry exists, add an entry to your output.

Do not add list entries if an entry already exists that matches that described by the user.

Rather than specifying observables, the user may ask you questions; respond to those questions as appropriate then ask the user again to describe the observables they want to compute. Use this information to populate your output."""


   parameter_rules = [
      """observables:

  You must analyze the user's input description to identify which observables the user wants to compute. If no existing entry exists, you must add a corresponding ObservableInfo instance for each distinct new observable to "observables" in your output.
      
  Rules:
  - Any given observable can appear only once, even if the user wants to compute it multiple times with different inputs.
  - Your list must include every observable explicitly mentioned, and only those observables. Do not invent observables, do not combine observables unless the user explicitly describes them as the same, and do not add details that are not explicitly provided by the user.
  - Do not ask the user if they want to specify any more observables.
  - Do not ask the user to confirm the list of observables
  """,
  
  "obs_type: Populate the 'obs_type' field with an object of type appropriate to the observable. If the user describes an observable that is not supported, you must describe to the user which observables you support and ask the user which ones they want.",

  "obs_tag: You must assign a unique tag/name to this observable."
    ]
   
   def validator(new_obs):
      existing_tags = [o.obs_tag for o in current_obs.observables]
      existing_meson_types = [o.obs_type.meson_type for o in current_obs.observables if o.obs_type.type == "meson2pt" ]
      valid = True
      reason = ""

      new_tags = []
      new_meson_types = []

      for o in new_obs.observables:
         if o.obs_tag in existing_tags:
            valid=False
            reason += f"\nObservable with tag {o.obs_tag} already exists"
         if o.obs_tag in new_tags:
            valid=False
            reason += f"\nDuplicate tag {o.obs_tag} in output"
         new_tags.append(o.obs_tag)

         if o.obs_type.type == "meson2pt":
            if o.obs_type.meson_type in existing_meson_types:
               valid=False
               reason += f"\nmeson2pt observable with meson_type {o.obs_type.meson_type} already exists"
            if o.obs_type.meson_type in new_meson_types:
               valid=False
               reason += f"\nDuplicate meson2pt observable with meson_type {o.obs_type.meson_type} in output"
            new_meson_types.append(o.obs_type.meson_type)

      return (valid, reason)


   new_obs = parameterAgent(model, ObservablesInfo, role, tools=[], tool_rules=[], parameter_rules=parameter_rules, input_messages=HumanMessage(user_input), validator=validator)

   updated_state = current_obs
   for o in new_obs.observables:
      updated_state.observables.append(o)

   state.observables = updated_state

   return new_obs
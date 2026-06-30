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
from langchain.agents import create_agent
from femtomeas.agent_common.common import *
from femtomeas.meas_config_agent.hadrons_xml import HadronsXML
from femtomeas.agent_common.python_output_agent import parameterAgent

class RBPrecCGsolver(BaseModel):
    """red-black preconditioned conjugate gradient (CG) solver"""
    type: Literal["RBPrecCG"] = "RBPrecCG"
    residual: float = Field(...,description="the solver tolerance, residual or stopping condition. Typical values are in the range 1e-6 to 1e-9")
    maxIteration: NonNegativeInt = Field(...,description="maximum number of solver iterations.")
    guesser: str = Field(..., description="guesser instance.")
    
    def setXML(self,name,action,xml):
        opt = xml.addModule(name,"MSolver::RBPrecCG")
        HadronsXML.setValues(opt, [ ("action",action), ("maxIteration", self.maxIteration), ("residual", self.residual), ("guesser", "") ])

    
class SolverConfig(BaseModel):
    name : str = Field(..., description="The name/tag for the solver instance")
    solver_args: Union[RBPrecCGsolver] = Field(..., description="Parameters of the solver. Each item must have a 'type' field. Valid values are: RBPrecCG", discriminator='type')
    action: str = Field(..., description="The name/tag of the action instance to use with the solver.")
    user_info: str = Field(..., description="Additional information (if any) provided by the user on what observables/propagators this solver will be used for")
    
    def setXML(self,xml):
        self.solver_args.setXML(self.name,self.action,xml)
    
def identifySolvers(model, state, user_interactions: list[BaseMessage]) -> str:
    """
    Parse the list of messages to identify a list of solver instances and their associated parameters
    """

    role = """identifying the solvers required for computing the lattice QCD propagators for the calculation.

Previous agent interactions have identified a set of observables and their required number of propagators. Solvers are required to compute those propagators. A solver instance has a set of parameters such as stopping conditions and the maximum number of iterations. The instance also has an 'action' field, that must be set to the name of one of the action instances identified previously. Each action instance must have one or more solver instances associated with it."""

    parameter_rules = [
  #solvers
        """solvers:
        
  Use the following workflow:
  1) If the user has not already done so in their previous responses, ask the user to specify what solver *types* they wish to use for which propagators. This question should not be specific to one observable or propagator; rather you should allow the user the freedom to specify information that could apply to multiple or even all propagators. In your question, list the solvers that you support but do not list their associated parameters. Do not ask the user to provide parameters at this stage.

     For example, "Specify the solvers required for the calculation (supported options: <OPTIONS>)."

     If there is only one supported solver you may assume this response and skip this question; however you must explain this to the user using the provideInformationToUser
        
  2) Instantiate an instance of SolverConfig for each required solver according to the rules described below.     

  Solver instance rules:
  - A different instance is required for each unique set of solver parameters. Some examples are as follows:
        a) If the user desires both a "sloppy" (loose tolerance) propagator and an "exact" (tight tolerance) propagator for a given action, create two solver instances with the same action but different residuals. This example is appropriate for an AMA style calculation.
        b) If the user specified the RBPrecCG solver type and there are action instances with names "action_1" and "action_2", create two separate solver instances with different values for the 'action' parameter.
    Note that these examples are just some of many possible workflows. Do not assume that the user desires either of these patterns. In particular, do not confuse the user by mentioning the concepts of sloppy or exact solvers unless the user has indicated that they want to do an AMA workflow.

  - Ensure there is at least one solver instance per action instance. 
  - Create a separate entry for each solver instance, even if the solver appears multiple times with different parameters.
  - Your list must include every solver instance explicitly mentioned, and only those. Do not invent instances. do not combine instances unless the user explicitly describes them as the same.""",
  #action
      """SolverConfig.action: Enter the name of the action associated with this solver instance. Each action should have one or more solvers.
  - Infer the action name from those associated with the propagators that the user describes
  - If there is only one action instance, do not ask the user which action to use.
      """,

  #name
      """SolverConfig.name: You must choose a unique tag/name to the solver instance. Assign this automatically, never ask the user (although they may choose to suggest names if they desire).
  - Never use the same tag for different instances.
  - The tag should include the action name and enough of the parameter values to uniquely distinguish it among the other solver instances, prefering shorter tags if possible.""",

  #user_info
      """SolverConfig.user_info: You must summarize any information relevant to what observables/propagators this solver will be used for provided by the user.
  - NEVER ask the user to specify this parameter. NEVER ask the user for additional information.
  - It is important that any positional information about the propagator be included, for example whether it is the first or second propagator of a two-point function, or if it is a 'spectator' quark in a baryon.
  - If the user does not specify any details, use an empty string.
  - For example, if the user specifies that this solver will be used for light quark propagators, enter "use for all light quark propagators" in user_info.""",

  #guesser
      """RBPrecCGsolver.guesser: This parameter currently supports using previously-computed eigenvectors to accelerate the solver

      You must use the following workflow:
   1) Check if any eigensolver instance exists with the same action name as this solver instance.
   2) If no, set guesser to an empty string and terminate this workflow.
      If yes,
      2a) ask the user to confirm whether to use these specific eigenvectors for the guesser parameter of this solver.
      2b) if they confirm, use the eigensolver's "name" parameter for the "guesser" parameter.
          if they do not confirm, use an empty string.
   Do not ask the user in general whether they would like to use eigenvectors if available. Only ask them to confirm the use of a specific set of eigenvectors for a specific solver."""
      
      ]

    additional_user_query_rules = [
        "When asking for parameters, phrase your questions to refer to groups of propagators that share the same partial set of parameters rather than specific propagators.",
        "Do not phrase questions specifically referring to action instances; all questions should refer to groups of observables and/or propagators.",
        "Do not assume that the solvers associated with propagators in these groups will all have the same parameters.",
        "Do not ask questions to the user that explicitly state that one value can be provided that applies to all instances in the group. If the user wants to specify a parameter that applies to more than one propagator in the group they will do so explicitly.",
        """Use plurals for parameter names associated with groups containing more than one propagator
  Examples:
    "Provide the stopping conditions for the solvers associated with the pion's propagators" (plural)
    "Provide the maximum number of iterations for the solver associated with the pion's first propagator" (singular)""",
    "When asking a question referring to a group, ensure your question clearly identifies the group."
    ]

    return parameterAgent(model, SolverConfig, "solvers", role, tools=[], tool_rules=[], parameter_rules=parameter_rules, input_messages=user_interactions, additional_user_query_rules=additional_user_query_rules)

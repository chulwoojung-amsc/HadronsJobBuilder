from pydantic import BaseModel, Field, ConfigDict, NonNegativeInt, TypeAdapter
from typing import Literal, Union, List, Optional, Tuple, ClassVar
from femtomeas.agent_common.common import *
from .hadrons_xml import HadronsXML
from femtomeas.agent_common.python_update_agent import parameterAgent, InstanceInfo
from femtomeas.agent_common.python_output_agent import executeCodeAndParse
from .solver_config_models import SolverConfig, RBPrecCGsolver
from .state import State
from .agent_workflows import checkValidNewGroupName, getCurrentState, startWorkflowMessage
from .agent_workflow_globals import registerWorkflowOperation, getUniqueIdx
from .agent_workflow_base import BaseRoutingHandle, BaseGroup
from femtomeas.agent_common.callgraph import Node
from .action_config import ActionGroup, ActionGroupHandle
from .eigenvectors import EigenSolverGroup, EigenSolverGroupHandle

def identifySolvers(model, group_name: str, action_group_name: str,  eigensolver_group_name : str | None, state: State, extra_info: str = ""):    
    action_group_code = state.groups[action_group_name].code

    use_evecs = eigensolver_group_name is not None

    #Guesser. Note: an empty string "" is always a valid guesser and means "no
    #guesser". Always explain this so the field is answerable even when no
    #eigensolver group is associated with the solver.
    guesser_directions = """
- The RBPrecCG solver's 'guesser' field takes an eigenvector-based initial guess to
  accelerate the solve. Using no guesser is allowed: to request that, set the
  guesser to the empty string ""."""

    if use_evecs:
        guesser_directions += f"""
  Alternatively, previously-computed eigenvectors can be used as guessers. The
  following eigensolver instances are available for the solvers you create:
  {state.groups[eigensolver_group_name].code}

  (The complete set of eigensolver instances is defined by this Python code:
  {state.instances["eigensolvers"]})

  - A valid guesser is either the empty string "" (no guesser) or one of the
    eigensolver instance names listed above. A named guesser must have the same
    action instance as the solver; for mixed-precision solvers the action
    precision must match."""

    ###############

    #Mixed-precision guidance: the MixedPrecisionRBPrecCG solver needs a
    #single-precision inner action in addition to the double-precision outer action.
    action_precisions_code = state.instances["actions"]
    mixed_precision_directions = f"""
- If the user chooses the MixedPrecisionRBPrecCG solver, it needs TWO action instances:
  the outer 'action' must be Double precision, and 'innerAction' must be a Single-precision
  action with the same physical parameters (mass, Ls, M5, etc.). The inner solve runs in
  single precision, the outer refinement in double.
  - The available action instances and their precisions are defined by the code above.
    If the user wants mixed precision but NO Single-precision action exists, you must tell
    them that a mixed-precision solver requires a single-precision action instance, and that
    they need to first create a Single-precision version of the action (same parameters,
    precision Single) in the action group. Do not fabricate an innerAction that does not exist."""


    role = f"""identifying the lattice QCD solver instances required by user. Solvers invert the QCD Dirac operator for a particular action instance. A solver instance has a set of parameters such as stopping conditions and the maximum number of iterations. The instance also has an 'action' field, that must be set to the name of one of the action instances identified previously.

- Your job is to help the user choose the solver and its parameters for one or more actions in the following subset:
{action_group_code}

where the parameters of the actions are defined through the following Python code:
{state.instances["actions"]}

- Ask for the solver type before asking about or mentioning the parameters of that solver. If there is only one supported solver you may assume this response and skip this question; however you must explain this to the user.

{guesser_directions}
{mixed_precision_directions}
-----------------
Solver instance rules
-----------------
You must adhere to the following rules for generating solver instances:
  - A different instance is required for each unique set of solver parameters. Some examples are as follows:
        a) If the user desires both a "sloppy" (loose tolerance) propagator and an "exact" (tight tolerance) propagator for a given action, create two solver instances with the same action but different residuals. This example is appropriate for an AMA style calculation.
        b) If the user specified the RBPrecCG solver type and there are action instances with names "action_1" and "action_2", create two separate solver instances with different values for the 'action' parameter.
    Note that these examples are just some of many possible workflows. Do not assume that the user desires either of these patterns. In particular, do not confuse the user by mentioning the concepts of sloppy or exact solvers unless the user has indicated that they want to do an AMA workflow.

  - Solver instances must each have unique parameters. Only instantiate a new solver instance if one with the required parameters does not already exist.
  - Solver instances should be reused wherever possible. Do not insist that each action have a unique solver.
  - Your list must include every solver instance explicitly mentioned, and only those. Do not invent instances. do not combine instances unless the user explicitly describes them as the same.
"""

    parameter_rules = [
  #action
      """SolverConfig.action: Enter the name of the action associated with this solver instance. 
  - This name must be within the provided subset of action names.
  - This name must correspond to an action with precision "Double"
  - If there is only one action instance, do not ask the user which action to use.
      """,

  #name
      """SolverConfig.name: You must choose a unique tag/name to the solver instance. Assign this automatically, never ask the user (although they may choose to suggest names if they desire).
  - Never use the same tag for different instances.
  - The tag should include the action name and enough of the parameter values to uniquely distinguish it among the other solver instances, prefering shorter tags if possible.""",

  #guesser
      """guesser fields (RBPrecCGsolver.guesser, and MixedPrecisionRBPrecCG's innerGuesser/outerGuesser): When you ask the user for any guesser, you MUST list the valid choices in your question so they know what to pick. The valid choices are:
  - "no guesser" (recorded as an empty string ""), which is always available, AND
  - any of the eigensolver instance names made available in the role instructions above (only if an eigensolver group is associated with this solver).
  For the mixed-precision solver, innerGuesser must be a single-precision eigensolver or "" and outerGuesser must be a double-precision eigensolver or "".
  If no eigensolver instances are available, tell the user that only "no guesser" is possible and record the empty string "" without making them guess. Never ask for a guesser without presenting these options, and never accept a value that is not one of the listed names or the empty string."""
      ]

    additional_user_query_rules = [        
    ]

    user_info_rules = """- You must use an empty string for this parameter."""

    used_actions = [ a.instance_tag for a in state.getGroup(action_group_name) ]

    used_eigsol = [""]
    if use_evecs:        
        used_eigsol = used_eigsol + [ a.instance_tag for a in state.getGroup(eigensolver_group_name) ]

    def checkAll(solvers):
        for i in range(len(solvers)):
            for j in range(i+1, len(solvers)):
                if solvers[i].action == solvers[j].action and solvers[i].solver_args == solvers[j].solver_args:
                    return (False, f"Action instances {solvers[i].name} and {solvers[j].name} have the same parameters. Solver instances must be unique.")                    

        #Check it only used actions previously described as being associated with this observable (note, this validator is only applied to *new* instances created by the agent)
        for s in solvers:
            if s.action not in used_actions:
                return (False, f"Action {s.action} is not within the provided subset of actions associated with this group")
            if isinstance(s.solver_args, RBPrecCGsolver) and s.solver_args.guesser not in used_eigsol:
                return (False, f"Guesser {s.solver_args.guesser} is not within the provided subset of eigensolvers associated with this group")
            
        return (True, "")
    ###################################
    
    inst = state.getInstanceCode("solvers")

    updated_solver_code, group_solver_code, _ = parameterAgent(model, SolverConfig, "solvers", inst.value, group_name, None, role, tools=[], \
                                                            parameter_rules=parameter_rules, user_info_rules=user_info_rules, group_validator=checkAll, instance_validator=lambda a: a.check(state), additional_user_query_rules=additional_user_query_rules,
                                                            input_messages=startWorkflowMessage(extra_info))
    inst.value = updated_solver_code
    return group_solver_code
    

class SolverGroupHandle(BaseRoutingHandle):
    pass

class SolverGroup(BaseGroup):
    handle_type : ClassVar[type] = SolverGroupHandle
    code: str = Field(..., description="Code for generating the list of solver instances in the group")

@registerWorkflowOperation(instance_info=("solvers", SolverConfig) )       
def createSolverGroup(group_name: str, actions: ActionGroupHandle, eigensolver: None | EigenSolverGroupHandle = None, *, user_info: str = "")->SolverGroupHandle:
    """Create a group of propagator solver module instances. These compute the inverse of the action's Dirac operator. Eigenvectors can be used to speed up the inversion, if available.

user_info: if specified by the user, provide the solver types or any other parameters related to the solvers.
    """

    checkValidNewGroupName(group_name)
    def doit(group_name, gactions_group_name: str, geigensolver_group_name : str | None):
        state, llm_model = getCurrentState()
        assert gactions_group_name in state.groups and isinstance(state.groups[gactions_group_name], ActionGroup)

        if geigensolver_group_name is not None:
            assert geigensolver_group_name in state.groups and isinstance(state.groups[geigensolver_group_name], EigenSolverGroup)

        state.groups[group_name] = SolverGroup(code = identifySolvers(llm_model, group_name, gactions_group_name, geigensolver_group_name, state, user_info ))   
        print("createSolverGroup: ", state.groups[group_name].code,  "\nSolvers is now: ", state.instances["solvers"])   
        return group_name

    if eigensolver is None:
        return SolverGroupHandle(Node(f"createSolverGroup_{getUniqueIdx()}", lambda gactions: doit(group_name, gactions, None), input_deps=[actions] ) )
    else:
        return SolverGroupHandle(Node(f"createSolverGroup_{getUniqueIdx()}", lambda gactions, geigens: doit(group_name, gactions, geigens), input_deps=[actions, eigensolver] ) )



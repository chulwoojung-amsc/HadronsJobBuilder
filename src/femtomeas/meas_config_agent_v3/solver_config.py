from pydantic import BaseModel, Field, ConfigDict, NonNegativeInt, TypeAdapter
from typing import Literal, Union, List, Optional, Tuple, ClassVar
from langchain.agents.structured_output import ToolStrategy, ProviderStrategy
from langchain.agents import create_agent
from femtomeas.agent_common.common import *
from femtomeas.meas_config_agent.hadrons_xml import HadronsXML
from femtomeas.agent_common.python_update_agent import parameterAgent, InstanceInfo
from femtomeas.agent_common.python_output_agent import executeCodeAndParse
from femtomeas.meas_config_agent_v2.solver_config_models import SolverConfig, RBPrecCGsolver
from .state import State
from .agent_workflows import BaseGroup, BaseGroupHandle, registerWorkflowOperation, checkValidNewGroupName, getUniqueIdx, addReservedName, getCurrentState
from femtomeas.agent_common.callgraph import Node
from .action_config import ActionGroup, ActionGroupHandle
from .eigenvectors import EigenSolverGroup, EigenSolverGroupHandle
    
def identifySolvers(model, group_name: str, action_group_name: str,  eigensolver_group_name : str | None, state: State):    
    action_group_code = state.groups[action_group_name].code

    use_evecs = eigensolver_group_name is not None

    #Guesser
    guesser_directions = """RBPrecCGsolver.guesser: Set this parameter to an empty string."""

    if use_evecs:
        guesser_directions = f"""RBPrecCGsolver.guesser: This parameter currently supports using previously-computed eigenvectors to accelerate the solver

      You must use the following workflow:
   1) Check if any eigensolver instance within the following list exists with the same action name as this solver instance:
      {state.groups[eigensolver_group_name].code}
 
      Where the complete set of eigensolver instances is defined through the following Python code:
      {state.eigensolvers}
   
   2) If no, set guesser to an empty string and terminate this workflow.
      If yes,
      2a) ask the user to confirm whether to use these specific eigenvectors for the guesser parameter of this solver.
      2b) if they confirm, use the eigensolver's "name" parameter for the "guesser" parameter.
          if they do not confirm, use an empty string.
   Do not ask the user in general whether they would like to use eigenvectors if available. Only ask them to confirm the use of a specific set of eigenvectors for a specific solver."""

    ###############


    role = f"""identifying the lattice QCD solver instances required by user. Solvers invert the QCD Dirac operator for a particular action instance. A solver instance has a set of parameters such as stopping conditions and the maximum number of iterations. The instance also has an 'action' field, that must be set to the name of one of the action instances identified previously.

- Your job is to help the user choose the solver and its parameters for one or more actions in the following subset:
{action_group_code}

where the parameters of the actions are defined through the following Python code:
{state.actions}

- Ask for the solver type before asking about or mentioning the parameters of that solver. If there is only one supported solver you may assume this response and skip this question; however you must explain this to the user.

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
      """SolverConfig.action: Enter the name of the action associated with this solver instance. This name must be within the provided subset of action names.
  - If there is only one action instance, do not ask the user which action to use.
      """,

  #name
      """SolverConfig.name: You must choose a unique tag/name to the solver instance. Assign this automatically, never ask the user (although they may choose to suggest names if they desire).
  - Never use the same tag for different instances.
  - The tag should include the action name and enough of the parameter values to uniquely distinguish it among the other solver instances, prefering shorter tags if possible.""",

  #guesser
      guesser_directions 
      ]

    additional_user_query_rules = [        
    ]

    user_info_rules = """- You must use an empty string for this parameter."""


    act, _ = executeCodeAndParse(action_group_code, InstanceInfo, action_group_name) 
    used_actions = [ a.instance_tag for a in act ]

    used_eigsol = [""]
    if use_evecs:
        esol, _ = executeCodeAndParse(state.groups[eigensolver_group_name].code, InstanceInfo, eigensolver_group_name) 
        used_eigsol = used_eigsol + [ a.instance_tag for a in esol ]

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
                return (False, f"Action {s.guesser} is not within the provided subset of eigensolvers associated with this group")
            
        return (True, "")
    ###################################

    print("SOLVERCONFIG ROLE", role)

    updated_solver_code, group_solver_code, _ = parameterAgent(model, SolverConfig, "solvers", state.solvers, group_name, None, role, tools=[], \
                                                            parameter_rules=parameter_rules, user_info_rules=user_info_rules, group_validator=checkAll, additional_user_query_rules=additional_user_query_rules)
    state.solvers = updated_solver_code
    return group_solver_code
    

class SolverGroupHandle(BaseGroupHandle):
    pass

class SolverGroup(BaseGroup):
    handle_type : ClassVar[type] = SolverGroupHandle
    code: str = Field(..., description="Code for generating the list of solver instances in the group")
   
def createSolverGroup(group_name: str, actions: ActionGroupHandle, eigensolver: None | EigenSolverGroupHandle = None)->SolverGroupHandle:
    checkValidNewGroupName(group_name)
    def doit(group_name, gactions_group_name: str, geigensolver_group_name : str | None):
        state, llm_model = getCurrentState()
        assert gactions_group_name in state.groups and isinstance(state.groups[gactions_group_name], ActionGroup)

        if geigensolver_group_name is not None:
            assert geigensolver_group_name in state.groups and isinstance(state.groups[geigensolver_group_name], EigenSolverGroup)

        state.groups[group_name] = SolverGroup(code = identifySolvers(llm_model, group_name, gactions_group_name, geigensolver_group_name, state ))   
        print("createSolverGroup: ", state.groups[group_name].code,  "\nSolvers is now: ", state.solvers)   
        return group_name

    if eigensolver is None:
        return SolverGroupHandle(group_name, Node(f"createSolverGroup_{getUniqueIdx()}", lambda gactions: doit(group_name, gactions, None), input_deps=[actions] ) )
    else:
        return SolverGroupHandle(group_name, Node(f"createSolverGroup_{getUniqueIdx()}", lambda gactions, geigens: doit(group_name, gactions, geigens), input_deps=[actions, eigensolver] ) )

registerWorkflowOperation(createSolverGroup)
addReservedName("solvers")
import os
from femtomeas.agent_common.common import *
from .state import *
from femtomeas.meas_config_agent.hadrons_xml import HadronsXML
import femtomeas.workflow_manager as wfman
from .action_config import identifyActions
from .observable_info import observableSkills, identifyObservables
from .source_config import identifySources
from .eigenvectors import setupEigenSolvers
from .solver_config import identifySolvers
from .propagator_config import identifyPropagators
from .smeared_prop_config import identifySmearedPropagators
from .observable_config import configureObservables
from femtomeas.meas_config_agent.gauge import identifyGaugeConfigs

def agent(query, model, ckpoint_file="state.json", reload_state=False)-> State :
    if reload_state and os.path.exists(ckpoint_file):
        state = reloadStateCheckpoint(ckpoint_file)
        query = state.query
        print("Reloaded query from state file:", query)
    else:
        state = State()

    state.query = query
    messages = [HumanMessage(query)]
        
    print(state)

    if state.observables == None:
        Print("""
---        
## IDENTIFY OBSERVABLES
---
        """)       
        state.observables = identifyObservables(model, messages.copy()).observables
        checkpointState(state,ckpoint_file)

    #Augment messages with information derived from observables
    messages.append( HumanMessage("The following information has been derived regarding the observables we need to compute based on user input:\n" + json.dumps(TypeAdapter(List[ObservableInfo]).dump_python(state.observables)  , indent=2) + "\n" + observableSkills(state.observables) ) )

    if state.actions == None:
       Print("""
---
## ACTIONS
---
       """)       
       state.actions = identifyActions(model, messages.copy())
       checkpointState(state,ckpoint_file)

    #Add actions information to messages
    #TODO: We don't need to pass forward all the parameter details of the actions, only the user_info and instance names are required
    messages.append( HumanMessage("The Python code that specifies and generates the action instances is as follows:\n" + state.actions, indent=2) )     

    if state.sources == None:
        Print("""
---
## SOURCES
---
        """)
        state.sources = identifySources(model, state, messages.copy())
        checkpointState(state,ckpoint_file)


    if state.eigensolvers == None:
        Print("""
---
## EIGENSOLVERS
---
        """) 
        state.eigensolvers = setupEigenSolvers(model, state, messages.copy())
        checkpointState(state,ckpoint_file)

    #Add eigensolvers to messages
    messages.append( HumanMessage("The Python code that specifies and generates the eigensolver instances is as follows:\n" + state.eigensolvers, indent=2) )
        
    if state.solvers == None:
        Print("""
---
## SOLVERS
---
        """) 
        state.solvers = identifySolvers(model, state, messages.copy())
        checkpointState(state,ckpoint_file)

    #Add sources and solvers to messages
    messages.append( HumanMessage("The Python code that specifies and generates the source instances is as follows:\n" + state.sources, indent=2) ) 
    messages.append( HumanMessage("The Python code that specifies and generates the solver instances is as follows:\n" + state.solvers, indent=2) ) 

    if state.propagators == None:
        Print("""
---
## PROPAGATORS
---
        """) 
        state.propagators = identifyPropagators(model, state, messages.copy())
        checkpointState(state,ckpoint_file)

    messages.append( HumanMessage("The Python code that specifies and generates the propagator instances is as follows:\n" + state.propagators, indent=2) )

    if state.smeared_propagators == None:
        Print("""
---
## SMEARED PROPAGATORS
---
        """) 
        state.smeared_propagators = identifySmearedPropagators(model, state, messages.copy())
        checkpointState(state,ckpoint_file)

    messages.append( HumanMessage("The Python code that specifies and generates the smeared propagator instances is as follows:\n" + state.smeared_propagators, indent=2) )


    if state.observable_configs == None:
                
        Print("""
---
## OBSERVABLE CONFIGURATIONS
---
        """)
        state.observable_configs = configureObservables(model, state, messages.copy())
        checkpointState(state,ckpoint_file)

    if state.gauge == None:
        Print("""
---
## GAUGE CONFIGURATIONS
---
        """)
        state.gauge = identifyGaugeConfigs(model, messages.copy())
        print("CHECKPOINTING STATE")
        checkpointState(state,ckpoint_file)
        print("CHECKPOINTING STATE COMPLETE")


    print("MEASUREMENT AGENT COMPLETE")
    return state

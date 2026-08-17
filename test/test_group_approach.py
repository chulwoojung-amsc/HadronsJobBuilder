from femtomeas.meas_config_agent.agent_workflows import *
from femtomeas.meas_config_agent.agent import *
from femtomeas.meas_config_agent.action_config_models import DWFaction, ActionConfig
from femtomeas.agent_common.python_update_agent import InstanceInfo
from femtomeas.meas_config_agent.source_config_models import SourceConfig, PointSource, WallSource
from femtomeas.meas_config_agent.solver_config_models import RBPrecCGsolver, SolverConfig
from femtomeas.meas_config_agent.propagator_config_models import PropagatorConfig
from femtomeas.meas_config_agent.smeared_prop_config_models import SmearedPropagatorConfig, WallSmear
from femtomeas.meas_config_agent.eigenvectors_models import EigenSolverConfig, LanczosEigenSolver, ChebyParams

from femtomeas.meas_config_agent.action_config import createActionGroup, ActionGroup
from femtomeas.meas_config_agent.solver_config import createSolverGroup, SolverGroup
from femtomeas.meas_config_agent.source_config import createSourceGroup, SourceGroup
from femtomeas.meas_config_agent.propagator_config import createPropagatorGroup, PropagatorGroup
from femtomeas.meas_config_agent.observable_config import createMeson2ptGroup
from femtomeas.meas_config_agent.smeared_prop_config import createSmearedPropagatorGroup, SmearedPropagatorGroup
from femtomeas.meas_config_agent.eigenvectors import createEigenSolverGroup, EigenSolverGroup

from femtomeas.meas_config_agent.state import State, reloadStateCheckpoint, checkpointState
import os
from femtomeas.meas_config_agent.gauge import GaugeFieldConfig, UnitGauge, identifyGaugeConfigs
import io
from langchain_openai import ChatOpenAI
import sys

def encodeInstances(list_name, inst):
    if not isinstance(inst, list):
        return encodeInstances(list_name, [inst])
    return f"{list_name} = [" + ", ".join([ str(a.model_dump()) for a in inst ] ) + "]"    

def encodeGroup(list_name, names):
    if not isinstance(names, list):
        return encodeGroup(list_name, [names])
    return encodeInstances(list_name, [ InstanceInfo(instance_tag=n, user_info="") for n in names ])


if __name__ == "__main__":
    if len(sys.argv) == 1:
        print("Need param file")

    config = readManagerConfigFile(sys.argv[1])

    amsc_llm_0t = ChatOpenAI(
        model="gpt-oss-120b",
        base_url="https://api.i2-core.american-science-cloud.org/",
        temperature=0,
        api_key = readI2APIkey(config.agent.i2api_key_path)
    )

    if 0:
        #Test the function manifest includes all the agent actions
        print(workflowFunctionManifest())

    if 0:
        #Test state serialization
        state = State()
        state.instances["actions"] = encodeInstances("actions", ActionConfig(name="action_inst", precision="Double", action=DWFaction(Ls=12, mass=0.01, M5=1.8) ) )
        state.groups["agroup1"] = ActionGroup(code = encodeGroup("agroup1", "action_inst"))

        state.instances["sources"] = encodeInstances("sources", [ SourceConfig(name="wall_0", source=WallSource(timeslice=0, momentum=(0,0,0,0)) ),  SourceConfig(name="point_0", source=PointSource(location=(0,0,0,0)) )   ])            
        state.groups["source_group"] = SourceGroup(code = encodeGroup("source_group", ["wall_0", "point_0"]) )

        fn = "test_state.json"
        if os.path.exists(fn):
            os.remove(fn)
        checkpointState(state, fn)

        rstate = reloadStateCheckpoint(fn)

        assert "actions" in rstate.instances
        assert rstate.instances["actions"] == state.instances["actions"]
        assert "agroup1" in rstate.groups and isinstance(rstate.groups["agroup1"], ActionGroup)
        assert rstate.groups["agroup1"].code == state.groups["agroup1"].code

        assert "sources" in rstate.instances
        assert rstate.instances["sources"] == state.instances["sources"]
        assert "source_group" in rstate.groups and isinstance(rstate.groups["source_group"], SourceGroup)
        assert rstate.groups["source_group"].code == state.groups["source_group"].code


    if 0:
        #Test instance caching
        state = State()
        ainst = ActionConfig(name="action_inst", precision="Double", action=DWFaction(Ls=12, mass=0.01, M5=1.8) )
        state.instances["actions"] = encodeInstances("actions",  ainst)

        got = state.getInstance("actions", "action_inst")
        assert got == ainst

        got2 = state.getInstance("actions", "action_inst")
        assert got2 is got

        ainst2 = ActionConfig(name="action_inst", precision="Single", action=DWFaction(Ls=13, mass=0.01, M5=1.9) )
        state.instances["actions"] = encodeInstances("actions",  ainst2)

        got3 = state.getInstance("actions", "action_inst")
        assert got3 != ainst
        assert got3 == ainst2
        assert got3 is not got2

    if 0:
        #Test group caching
        state = State()        
        state.groups["agroup1"] = ActionGroup(code = encodeGroup("agroup1", "action_inst"))

        gp = state.getGroup("agroup1")
        assert len(gp) == 1 and gp[0].instance_tag == "action_inst"

        gp2 = state.getGroup("agroup1")
        assert gp2 is gp


    def testEnactor(func, args):
        print("ENACTING ",func.__name__, args)
        return func(*args)

    if 0:
        #Test solver agent
        state = State()
        state.instances["actions"] = encodeInstances("actions", ActionConfig(name="action_inst", precision="Double", action=DWFaction(Ls=12, mass=0.01, M5=1.8) ) )
        state.groups["agroup1"] = ActionGroup(code = encodeGroup("agroup1", "action_inst"))

        initializeState(amsc_llm_0t, state)

        ag1_h = retrieveGroupHandle("agroup1")
        sg1_h = createSolverGroup("sgroup1", ag1_h)
            
        graph = sg1_h.parent_node
        graph.eval(enactor=testEnactor)

    if 0:
        #Test mixed-prec solver agent
        state = State()
        state.instances["actions"] = encodeInstances("actions", [
            ActionConfig(name="action_d", precision="Double", action=DWFaction(Ls=12, mass=0.01, M5=1.8) ),
            ActionConfig(name="action_s", precision="Single", action=DWFaction(Ls=12, mass=0.01, M5=1.8) )
        ])
        state.groups["agroup1"] = ActionGroup(code = encodeGroup("agroup1", ["action_d", "action_s"]))

        initializeState(amsc_llm_0t, state)

        ag1_h = retrieveGroupHandle("agroup1")
        sg1_h = createSolverGroup("sgroup1", ag1_h)
            
        graph = sg1_h.parent_node
        graph.eval(enactor=testEnactor)        

    if 1:
        #Test mixed-prec solver agent with guessers
        state = State()    

        state.instances["actions"] = encodeInstances("actions", [
            ActionConfig(name="action_d", precision="Double", action=DWFaction(Ls=12, mass=0.01, M5=1.8) ),
            ActionConfig(name="action_s", precision="Single", action=DWFaction(Ls=12, mass=0.01, M5=1.8) )
        ])
        state.groups["agroup1"] = ActionGroup(code = encodeGroup("agroup1", ["action_d", "action_s"]))

        state.instances["eigensolvers"] = encodeInstances("eigensolvers",  [
            EigenSolverConfig(name="esol_d", action="action_d", solver_args=LanczosEigenSolver(cheby=ChebyParams(alpha=0.01,beta=3.2,Npoly=101), Nstop=100, Nk=100, Nextra=10, resid=1e-7, MaxIt=20, storeEvecs=False, fileStem="" )),
            EigenSolverConfig(name="esol_s", action="action_s", solver_args=LanczosEigenSolver(cheby=ChebyParams(alpha=0.01,beta=3.2,Npoly=101), Nstop=100, Nk=100, Nextra=10, resid=1e-7, MaxIt=20, storeEvecs=False, fileStem="" ))
            ])
        state.groups["egroup1"] = EigenSolverGroup(code = encodeGroup("egroup1", ["esol_d", "esol_s"]))

        initializeState(amsc_llm_0t, state)

        ag1_h = retrieveGroupHandle("agroup1")        
        eg1_h = retrieveGroupHandle("egroup1")
        sg1_h = createSolverGroup("sgroup1", ag1_h, eg1_h)
            
        graph = sg1_h.parent_node
        graph.eval(enactor=testEnactor)        


    if 0:
        #Test handle-type check when graph is created
        state = State()    

        state.instances["sources"] = encodeInstances("sources", SourceConfig(name="wall_0", source=WallSource(timeslice=0, momentum=(0,0,0,0)) ) )            
        state.groups["source_group"] = SourceGroup(code = encodeGroup("source_group", ["wall_0", "point_0"]) )

        state.instances["actions"] = encodeInstances("actions", ActionConfig(name="action_d", precision="Double", action=DWFaction(Ls=12, mass=0.01, M5=1.8) ) )            
        state.groups["agroup1"] = ActionGroup(code = encodeGroup("agroup1", ["action_d", "action_s"]))

        state.instances["eigensolvers"] = encodeInstances("eigensolvers",  
            EigenSolverConfig(name="esol_d", action="action_d", solver_args=LanczosEigenSolver(cheby=ChebyParams(alpha=0.01,beta=3.2,Npoly=101), Nstop=100, Nk=100, Nextra=10, resid=1e-7, MaxIt=20, storeEvecs=False, fileStem="" )))
        state.groups["egroup1"] = EigenSolverGroup(code = encodeGroup("egroup1", ["esol_d", "esol_s"]))

        initializeState(amsc_llm_0t, state)

        ag1_h = retrieveGroupHandle("agroup1")        
        eg1_h = retrieveGroupHandle("egroup1")
        sg1_h = retrieveGroupHandle("source_group")

        #Try feeding a source group instead of an action group
        caught=False
        try:
            slg1_h = createSolverGroup("slgroup1", sg1_h, eg1_h)
        except Exception as e:
            print("CAUGHT EXPECTED EXCEPTION: ", e)
            caught=True
        assert caught

        #Eigensolver is an optional argument, check it works with None and a correct type
        caught=False
        try:
            slg1_h = createSolverGroup("slgroup1", ag1_h, eg1_h)
        except Exception as e:
            print("CAUGHT *UN*EXPECTED EXCEPTION: ", e)
            caught=True
        assert not caught

        caught=False
        try:
            slg1_h = createSolverGroup("slgroup1", ag1_h, None)
        except Exception as e:
            print("CAUGHT *UN*EXPECTED EXCEPTION: ", e)
            caught=True
        assert not caught

        #Now check it only accepts within the expected types
        caught=False
        try:
            slg1_h = createSolverGroup("slgroup1", ag1_h, sg1_h)
        except Exception as e:
            print("CAUGHT EXPECTED EXCEPTION: ", e)
            caught=True
        assert caught


    if 0:
        #Test gauge agent
        identifyGaugeConfigs(amsc_llm_0t, [HumanMessage("Start your workflow")])


    if 0:
        #Test source agent
        state = State()
        initializeState(amsc_llm_0t, state)

        sg1_h = createSourceGroup("sgroup1")
            
        graph = sg1_h.parent_node
        graph.eval(enactor=testEnactor)


    if 0:
        #Test action agent
        state = State()
        initializeState(amsc_llm_0t, state)

        sg1_h = createActionGroup("sgroup1")
            
        graph = sg1_h.parent_node
        graph.eval(enactor=testEnactor)

    if 0:
        #Test propagator agent
        state = State()
        state.instances["sources"] = encodeInstances("sources", [ SourceConfig(name="wall_0", source=WallSource(timeslice=0, momentum=(0,0,0,0)) ),  SourceConfig(name="point_0", source=PointSource(location=(0,0,0,0)) )   ])            
        state.groups["source_group"] = SourceGroup(code = encodeGroup("source_group", ["wall_0", "point_0"]) )
        state.instances["solvers"] = encodeInstances("solvers", [ SolverConfig(name="solv", solver_args=RBPrecCGsolver(residual=1e-8,maxIteration=10000,guesser=""), action="action_1")] )    
        state.groups["solver_group"] = SolverGroup(code = encodeGroup("solver_group", "solv") )

        initializeState(amsc_llm_0t, state)

        src_h = retrieveGroupHandle("source_group")
        slv_h = retrieveGroupHandle("solver_group")
        prop_h = createPropagatorGroup("prop1", src_h, slv_h)
            
        graph = prop_h.parent_node
        graph.eval(enactor=testEnactor)

    if 0:
        #Test meson2pt agent
        state = State()
        state.instances["propagators"] = encodeInstances("propagators",  [ PropagatorConfig(name="prop_wall_t32", source="wall_src_t32", solver="solver"),   PropagatorConfig(name="prop_wall_t0", source="wall_src_t0", solver="solver")     ]  )
        state.groups["prop_group"] = PropagatorGroup(code = encodeGroup("prop_group", ["prop_wall_t32", "prop_wall_t0"]) )

        initializeState(amsc_llm_0t, state)

        prop_h = retrieveGroupHandle("prop_group")
        meson_h = createMeson2ptGroup("meson2pt_group",prop_h)

        graph = meson_h.parent_node
        graph.eval(enactor=testEnactor)


    if 0:
        #Test smeared propagator agent
        state = State()
        
        state.instances["propagators"] = encodeInstances("propagators", [ PropagatorConfig(name="prop_wall_t32", source="wall_src_t32", solver="solver"),   PropagatorConfig(name="prop_wall_t0", source="wall_src_t0", solver="solver")     ])    
        state.groups["prop_group"] = PropagatorGroup(code = encodeGroup("prop_group", ["prop_wall_t32", "prop_wall_t0" ]) )

        initializeState(amsc_llm_0t, state)

        prop_h = retrieveGroupHandle("prop_group")
        sprop_h = createSmearedPropagatorGroup("sprop_group",prop_h)

        graph = sprop_h.parent_node
        graph.eval(enactor=testEnactor)


    if 0:
        #Test meson2pt agent with smeared props
        state = State()        
        state.instances["propagators"]  = encodeInstances("propagators", [ PropagatorConfig(name="prop_wall_t32", source="wall_src_t32", solver="solver"),   PropagatorConfig(name="prop_wall_t0", source="wall_src_t0", solver="solver")     ])        
        state.instances["smeared_propagators"] = encodeInstances("smeared_propagators", [ SmearedPropagatorConfig(name="sprop_w32", input_prop="prop_wall_t32", smearing=WallSmear(momentum=(0,0,0,0))),
                       SmearedPropagatorConfig(name="sprop_w0", input_prop="prop_wall_t0", smearing=WallSmear(momentum=(0,0,0,0))),
                         ])
        state.groups["sprop_group"] = SmearedPropagatorGroup(code = encodeGroup("sprop_group", ["sprop_w32", "sprop_w0"]) )
                                                             
        initializeState(amsc_llm_0t, state)

        prop_h = retrieveGroupHandle("sprop_group")
        meson_h = createMeson2ptGroup("meson2pt_group",prop_h)

        graph = meson_h.parent_node
        graph.eval(enactor=testEnactor)        


    if 0:
        #Test eigensolver agent
        state = State()        
        state.instances["actions"] = encodeInstances("actions", ActionConfig(name="action_inst", action=DWFaction(Ls=12, mass=0.01, M5=1.8) ))
        state.groups["agroup1"] = ActionGroup(code = encodeGroup("agroup1", "action_inst"))

        initializeState(amsc_llm_0t, state)

        ag1_h = retrieveGroupHandle("agroup1")
        sg1_h = createEigenSolverGroup("sgroup1", ag1_h)
            
        graph = sg1_h.parent_node
        graph.eval(enactor=testEnactor)        


    if 0:
        #Test solver agent with eigenvectors
        state = State()
        
        state.instances["actions"] = encodeInstances("actions", ActionConfig(name="action_inst", action=DWFaction(Ls=12, mass=0.01, M5=1.8) ))    
        state.groups["agroup1"] = ActionGroup(code = encodeGroup("agroup1", "action_inst") )
       
        state.instances["eigensolvers"] = encodeInstances("eigensolvers",  EigenSolverConfig(name="esol1", action="action_inst", solver_args=LanczosEigenSolver(cheby=ChebyParams(alpha=0.01,beta=3.2,Npoly=101), Nstop=100, Nk=100, Nextra=10, resid=1e-7, MaxIt=20, storeEvecs=False, fileStem="" ))  )
        state.groups["egroup1"] = EigenSolverGroup(code = encodeGroup("egroup1", "esol1"))

        initializeState(amsc_llm_0t, state)

        ag1_h = retrieveGroupHandle("agroup1")
        eg1_h = retrieveGroupHandle("egroup1")
        sg1_h = createSolverGroup("sgroup1", ag1_h, eg1_h)
            
        graph = sg1_h.parent_node
        graph.eval(enactor=testEnactor)        


    if 0:
        #Test XML output
        state = State()
        state.instances["actions"] = encodeInstances("actions", ActionConfig(name="action_inst", precision="Double", action=DWFaction(Ls=12, mass=0.01, M5=1.8) ))    
        state.gauge = GaugeFieldConfig(Lx=4,Ly=4,Lz=4,Lt=4,config=UnitGauge())
        xml = state.toHadronsXML()
        print(xml.toString())

    if 0:
        #Test code generator output
        state = State()
        state.instances["actions"] = encodeInstances("actions", ActionConfig(name="action_inst", action=DWFaction(Ls=12, mass=0.01, M5=1.8) ))    
        state.gauge = GaugeFieldConfig(Lx=4,Ly=4,Lz=4,Lt=4,config=UnitGauge())
        strm = io.StringIO()
        state.toXMLgeneratorCodeStream(strm)
        print(strm.getvalue())
        state.toXMLgeneratorCode("test.py")
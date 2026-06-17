import json
from pydantic import BaseModel, Field, ConfigDict, NonNegativeInt, TypeAdapter
from typing import Literal, Union, List, Optional, Tuple

from .observable_info import ObservableInfo
from femtomeas.meas_config_agent.gauge import GaugeFieldConfig
from femtomeas.meas_config_agent.hadrons_xml import HadronsXML
from femtomeas.agent_common.common import Print
from femtomeas.agent_common.python_output_agent import executeCode, executeCodeAndParse

from .action_config import ActionConfig
from .source_config import SourceConfig
from .solver_config import SolverConfig
from .propagator_config import PropagatorConfig
from .observable_config import ObservableConfig

def checkpointState(state, filename):
    #j = json.dumps({k: v.model_dump() for k, v in state.items()}, indent=2)
    j=json.loads(state.model_dump_json())
    with open(filename, 'w') as wr:
        wr.write(json.dumps(j,indent=2))

def reloadStateCheckpoint(filename):
    Print("Reloading state checkpoint from",filename)
    with open(filename, 'r') as rd:
        j = rd.read()
    return State.model_validate_json(j)

class State(BaseModel):
    query: str | None = Field(None,description="The original query")
    observables: List[ObservableInfo] | None = Field(None,description="The list of observables and associated relevant information")
    actions: str | None = Field(None,description="Code for generating action instances")
    sources: str | None = Field(None,description="Code for generating source instances")
    eigensolvers: str | None = Field(None,description="Code for generating eigensolver instances")
    solvers: str | None = Field(None,description="Code for generating the solver instances")
    propagators: str | None = Field(None,description="Code for generating the propagator instances")
    observable_configs : str | None = Field(None,description="Code for generating observable instances")
    gauge: GaugeFieldConfig | None = Field(None,description="The gauge configuration parameters")

    def isValidObservable(self, obs_name):
        for p in self.observables:
            if p.name == obs_name:
                return True
        return False

    def locateObservable(self, obs_name) -> ObservableInfo | None:
        for p in self.observables:
            if p.name == obs_name:
                return p
        return None
    
    def isValidAction(self, action_name):
        r, e = executeCode(self.actions)
        if len(e) > 0:
            raise Exception(f"Executing code gave the following exceptions: {e}")
        assert "result" in r.keys()
        actions = r["result"]

        for p in actions:
            if p["name"] == action_name:
                return True
        return False
    
    def isValidSource(self, source_name):
        r, e = executeCode(self.sources)
        if len(e) > 0:
            raise Exception(f"Executing code gave the following exceptions: {e}")
        assert "result" in r.keys()
        sources = r["result"]

        for p in sources:
            if p["name"] == source_name:
                return True
        return False

    def isValidSolver(self, solver_name):
        r, e = executeCode(self.solvers)
        if len(e) > 0:
            raise Exception(f"Executing code gave the following exceptions: {e}")
        assert "result" in r.keys()
        solvers = r["result"]

        for p in solvers:
            if p["name"] == solver_name:
                return True
        return False
    
    def isValidPropagator(self, propagator_name):
        r, e = executeCode(self.propagators)
        if len(e) > 0:
            raise Exception(f"Executing code gave the following exceptions: {e}")
        assert "result" in r.keys()
        propagators = r["result"]

        for p in propagators:
            if p["name"] == propagator_name:
                return True
        return False


    def _toHadronsXMLbase(self)->HadronsXML:
        """
        Set all elements bar the gauge module, which needs special treatment
        """
        xml = HadronsXML()
        xml.setRunID(1234) #What does this do?

        for c in [(self.actions, ActionConfig), (self.sources, SourceConfig), (self.solvers, SolverConfig), (self.propagators, PropagatorConfig), (self.observable_configs, ObservableConfig)]:            
            r, e = executeCodeAndParse(*c)
            if len(e) > 0:
                raise Exception(f"Executing code for type {c[1]} gave the following exceptions: {e}")
            for a in r:
                a.setXML(xml)

        #Temporary; add a zero-momentum point sink for two-point functions
        #TODO: Have the observables agent also construct sinks as needed
        snk = xml.addModule("point_sink_zerop", "MSink::ScalarPoint")
        HadronsXML.setValue(snk, "mom", "0. 0. 0.")
        
        return xml
    
    def toHadronsXML(self)->HadronsXML:
        xml=self._toHadronsXMLbase()
        self.gauge.setXML(xml)
        return xml

    def toHadronsXMLsingleConf(self,job_index,override_path = None ):
        """
        Output the XML just for a single configuration.
        job_index : The index of the entry in the range, i.e. 0 -> start, 1 -> start+step,  etc
        override_path : Replace the path in which the file resides, e.g. if it was moved prior to execution
        """
        xml=self._toHadronsXMLbase()
        self.gauge.setXMLsingle(xml,job_index,override_path)
        return xml

    

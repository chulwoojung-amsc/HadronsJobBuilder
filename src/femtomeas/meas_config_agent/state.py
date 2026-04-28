import json
from pydantic import BaseModel, Field, ConfigDict, NonNegativeInt, TypeAdapter
from typing import Literal, Union, List, Optional, Tuple

from .observable_info import *
from .observable_config import *
from .action_config import *
from .source_config import *
from .solver_config import *
from .propagator_config import *
from .gauge import *
from .eigenvectors import *
from .hadrons_xml import HadronsXML
from .common import Print

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
    actions: List[ActionConfig] | None = Field(None,description="The list of action instances")
    sources: List[SourceConfig] | None = Field(None,description="The list of source instances")
    eigensolvers: List[EigenSolverConfig] | None = Field(None,description="The list of eigensolver instances")
    solvers: List[SolverConfig] | None = Field(None,description="The list of solver instances")
    propagators: List[PropagatorConfig] | None = Field(None,description="The list of propagator instances")
    observable_configs : List[ObservableConfig] | None = Field(None,description="The list of observable instances")
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
        for p in self.actions:
            if p.name == action_name:
                return True
        return False
    
    def isValidSource(self, source_name):
        for p in self.sources:
            if p.name == source_name:
                return True
        return False

    def isValidSolver(self, solver_name):
        for p in self.solvers:
            if p.name == solver_name:
                return True
        return False   
    
    def isValidPropagator(self, prop_name):
        for p in self.propagators:
            if p.name == prop_name:
                return True
        return False

    def _toHadronsXMLbase(self)->HadronsXML:
        """
        Set all elements bar the gauge module, which needs special treatment
        """
        xml = HadronsXML()
        xml.setRunID(1234) #What does this do?

        for a in self.actions:
            a.setXML(xml)
        for s in self.sources:
            s.setXML(xml)
        for s in self.solvers:
            s.setXML(xml)
        for p in self.propagators:
            p.setXML(xml)

        #Temporary; add a zero-momentum point sink for two-point functions
        #TODO: Have the observables agent also construct sinks as needed
        snk = xml.addModule("point_sink_zerop", "MSink::ScalarPoint")
        HadronsXML.setValue(snk, "mom", "0. 0. 0.")

        for o in self.observable_configs:
            o.setXML(xml)
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

    

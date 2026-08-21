from femtomeas.meas_config_agent.hadrons_xml import HadronsXML
from .logging import wfmanLog
from dataclasses import dataclass
from typing import Tuple
from .actions_base import replaceJobIdSubstring, ComputeActionBase
import tempfile
from .hadrons import submitHadronsJob

class HadronsJobSpec:
    job_rundir : str #Job run directory. The <JOBID> substring will be replaced by the job index if present
    xml_spec : bytes #The HadronsXML spec as a bytestring
    grid : Tuple[int,int,int,int]

    def __init__(self, job_rundir, xml : HadronsXML, grid):
        self.job_rundir = job_rundir
        self.xml_spec = xml.toBytes()
        self.grid = grid

    def __repr__(self):
        xml = HadronsXML()
        xml.fromBytes(self.xml_spec)
        xmlstr = xml.toString()
        
        return f"HadronsJobSpec(job_rundir={self.job_rundir}, xml_spec={xmlstr}, grid={self.grid})"

    def writeXML(self, filename):
        xml = HadronsXML()
        xml.fromBytes(self.xml_spec)
        xml.write(filename)
        wfmanLog("XML written to",filename)
           
@dataclass
class HadronsComputeAction(ComputeActionBase):
    spec :  HadronsJobSpec
    mpi : Tuple[int, int, int, int]
   
    def initiateAction(self, job_id)->str:
        _, xml_file = tempfile.mkstemp(prefix="hadrons_xml_", text=True, dir="/tmp", suffix=".xml")
        self.spec.writeXML(xml_file)
        assert self.machine in globals.remote_workdir
        
        rundir = replaceJobIdSubstring(self.spec.job_rundir, job_id)
        wfmanLog(f"Job {job_id} machine {self.machine} rundir {rundir}")
        return submitHadronsJob(self.machine, xml_file, rundir, self.account, self.queue, self.time, self.spec.grid, self.mpi, delete_xml_after_upload = True)

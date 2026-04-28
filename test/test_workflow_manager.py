import os
os.environ["FEMTOMEAS_API_IMPL"] = "SPOOF"

import femtomeas.workflow_manager.globals as globals
from femtomeas.workflow_manager.api_general import setupWorkflowAgent
from femtomeas.workflow_manager.manager import *
from femtomeas.workflow_manager.hadrons import setHadronsInfo
from femtomeas.meas_config_agent.hadrons_xml import HadronsXML
from femtomeas.workflow_manager.manager_config import readManagerConfigFile
import time

setupWorkflowAgent("/path/to/sfapi_key", "/path/to/iriapi_key", { "Perlmutter" : "/path/to/sandbox" })
setHadronsInfo({ "Perlmutter" : { "bin" : "/global/u2/c/ckelly/CPS/install_mpi_pm_new/Hadrons_pm_new/bin",    "env" : "source /global/u2/c/ckelly/CPS/bld/grid_pm_develop/sourceme.sh" } } )

#readManagerConfigFile("man_config.json")

xml = HadronsXML()
xml.read("hadrons_run.xml")

# man = JobManager()
# man.start()

if 0:
    #Test active transfer progression
    jd = JobData()

    t1 = TransferToAction("dtn","/path/to/src","Perlmutter","/path/to/dest")
    t2 = TransferFromAction("Perlmutter","/path/to/src","dtn","/path/to/dest")
    jobid = jd.enqueueJob([t1,t2])

    jd.startWorkflows([jobid])
    status = jd.jobStatus(jobid)
    assert status['head_action_status'] == ActionStatus.ACTIVE

    #Track active transfer state until completion
    updates = {}
    while jobid not in updates:
        updates = jd.progressActiveActions(force_poll=True)
        time.sleep(2)
                
    print("Update action to status",updates[jobid])
    assert updates[jobid] == ActionStatus.COMPLETED

    #Enact next workflow stage
    jd.progressActiveWorkflows()
    status = jd.jobStatus(jobid)
    print("New action type", status['head_action_type'],"and status", status['head_action_status'], "expect", type(t2).__name__, "ACTIVE")
    assert status['head_action_type'] == type(t2).__name__ and status['head_action_status'] == ActionStatus.ACTIVE

    updates = {}
    while jobid not in updates:
        updates = jd.progressActiveActions(force_poll=True)
        time.sleep(2)

    print("Update action to status",updates[jobid])
    assert updates[jobid] == ActionStatus.COMPLETED

    jd.progressActiveWorkflows()
    status = jd.jobStatus(jobid)
    
    assert status['workflow_stage'] == 2 and status['head_action_status'] == ActionStatus.COMPLETED and status['head_action_class'] == ActionClass.NONE

if 0:
    #Test a complete workflow under a loop until termination
    jd = JobData()

    spec = HadronsJobSpec("/path/to/jobdir", xml, grid=(8,8,8,16) )
    t1 = TransferToAction("dtn","/path/to/src","Perlmutter","/path/to/dest")
    t2 = HadronsComputeAction(machine="Perlmutter",account="amsc013_g",queue="debug", time="300", spec=spec, mpi=(1,1,1,2) )
    t3 = TransferFromAction("Perlmutter","/path/to/src","dtn","/path/to/dest")
    jobid = jd.enqueueJob([t1,t2,t3])

    jd.startWorkflows([jobid])

    status = jd.jobStatus(jobid)
    while status['head_action_class'] != ActionClass.NONE:
        time.sleep(2)
        jd.progressActiveState(force_poll=2)
        status = jd.jobStatus(jobid)
    

if 1:
    #Test a complete workflow under the threaded loop
    jman = JobManager(poll_freq=1, max_workflows_active=0)  #max_workflows_active=0 -> manual control of job activation
    jman.start()
    
    spec = HadronsJobSpec("/path/to/sandbox/jobdir", xml, grid=(8,8,8,16) )
    t1 = TransferToAction("dtn","/path/to/src","Perlmutter","/path/to/sandbox/dest")
    t2 = HadronsComputeAction(machine="Perlmutter",account="amsc013_g",queue="debug", time="300", spec=spec, mpi=(1,1,1,2) )
    t3 = TransferFromAction("Perlmutter","/path/to/sandbox/src","dtn","/path/to/dest")

    with jman as jd:
        jobid = jd.enqueueJob([t1,t2,t3])
        jd.startWorkflows([jobid])
        status = jd.jobStatus(jobid)
    
    while status['head_action_class'] != ActionClass.NONE:
        time.sleep(2)
        status = jman(lambda jd: jd.jobStatus(jobid))
        transfer_status = jman(lambda jd: jd.getActiveActions(ActionClass.TRANSFER))
        print("ONGOING TRANSFERS:",transfer_status)
        
    jman.stop()


if 0:
    #Test workflow activation up to limit
    jman = JobManager(poll_freq=1)
    jman.start()
    
    spec = HadronsJobSpec("/path/to/jobdir", xml, grid=(8,8,8,16) )
    t1 = TransferToAction("dtn","/path/to/src","Perlmutter","/path/to/dest")
    t2 = HadronsComputeAction(machine="Perlmutter",account="amsc013_g",queue="debug", time="300", spec=spec, mpi=(1,1,1,2) )
    t3 = TransferFromAction("Perlmutter","/path/to/src","dtn","/path/to/dest")

    with jman as jd:
        jobid = jd.enqueueJob([t1,t2,t3])
    
    jman.stop()
    with jman as jd:
        status = jd.jobStatus(jobid)
        assert status['head_action_class'] == ActionClass.NONE

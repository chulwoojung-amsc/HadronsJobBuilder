import femtomeas.workflow_manager.globals as globals
globals.api_impl = "SPOOF"

from femtomeas.workflow_manager.api_general import setupWorkflowAgent
from femtomeas.workflow_manager.manager import *
from femtomeas.workflow_manager.hadrons import setHadronsInfo
from femtomeas.meas_config_agent.hadrons_xml import HadronsXML

import time

setupWorkflowAgent("/path/to/key", { "Perlmutter" : "/path/to/sandbox" })
setHadronsInfo({ "Perlmutter" : { "bin" : "/global/u2/c/ckelly/CPS/install_mpi_pm_new/Hadrons_pm_new/bin",    "env" : "source /global/u2/c/ckelly/CPS/bld/grid_pm_develop/sourceme.sh" } } )


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
    jman = JobManager(poll_freq=1)
    jman.start()
    
    spec = HadronsJobSpec("/path/to/jobdir", xml, grid=(8,8,8,16) )
    t1 = TransferToAction("dtn","/path/to/src","Perlmutter","/path/to/dest")
    t2 = HadronsComputeAction(machine="Perlmutter",account="amsc013_g",queue="debug", time="300", spec=spec, mpi=(1,1,1,2) )
    t3 = TransferFromAction("Perlmutter","/path/to/src","dtn","/path/to/dest")

    jobid = jman(lambda jd: jd.enqueueJob([t1,t2,t3]))
    jman(lambda jd: jd.startWorkflows([jobid]))
    
    status = jman(lambda jd: jd.jobStatus(jobid))
    while status['head_action_class'] != ActionClass.NONE:
        time.sleep(2)
        status = jman(lambda jd: jd.jobStatus(jobid))
    
    jman.stop()

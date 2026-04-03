from typing import Tuple
from femtomeas.meas_config_agent.state import State
from .manager import JobManager, TransferToAction, TransferFromAction, HadronsComputeAction, HadronsJobSpec
from . import globals

def enqueueStandardHadronsWorkflow(state : State, jman : JobManager,
                            mpi : Tuple[int,int,int,int], grid : Tuple[int,int,int,int],
                            machine : str, group_name : str,
                            account: str, queue : str, time : str):
    configs, source_uuid = state.gauge.getJobConfigurationsAndSource()
    
    if machine not in globals.remote_workdir:
        raise Exception(f"Unknown machine {machine}")
    
    job_dir = globals.remote_workdir[machine] + f"/{group_name}/<JOBID>"
    cfg_staging_dir = globals.remote_workdir[machine] + f"/{group_name}/configurations"

    print("enqueueStandardHadronsWorkflow is queueing",len(configs),"configurations:", configs)
    for i in range(len(configs)):
        workflow = []

        #If the configs are remote they will need to staged in
        override_cfgpath = None
        if source_uuid != None and configs[i] != None:
            action = TransferToAction(source_endpoint=source_uuid, source_path=configs[i], machine=machine, dest_path=cfg_staging_dir)
            workflow.append(action)
            override_cfgpath = cfg_staging_dir

        xml = state.toHadronsXMLsingleConf(i, override_path = override_cfgpath)
        spec = HadronsJobSpec(job_rundir=job_dir, xml=xml, grid=grid)

        workflow.append(
            HadronsComputeAction(machine=machine, account=account, queue=queue, time=time, spec=spec, mpi=mpi)
            )

        #Todo: stage out

        with jman as jd:
            jd.enqueueJob(workflow, group_name)

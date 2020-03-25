import os
import sys
import traci
import traci.constants as tc
import sumolib
import logging

# recommend reading in this order:
# scenario file ----> simulation manager ---> vehicle ---> platoon ---> intersection controller

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from src.simulationmanager import SimulationManager
from src.simlib import setUpSimulation
import src.BaseDefine as BaseDefine

platoonFlag = True
setUpSimulation("../maps/NormalIntersection_no_TLS_simple/NormalIntersection_no_TLS.sumocfg",1,platoonFlag)

step = 0
manager = SimulationManager(True, True, False)

# set junction listener
# junction in the map: gnej15
junctionID = 'gneJ15'
traci.junction.subscribeContext(junctionID, tc.CMD_GET_VEHICLE_VARIABLE, BaseDefine.JunctionDetectionRange,
                                [tc.VAR_SPEED, tc.VAR_WAITING_TIME])
print(traci.junction.getContextSubscriptionResults(junctionID))

while step < BaseDefine.MaxStep:
    manager.handleSimulationStep()
    traci.simulationStep()
    # monitor output
    # oux = traci.junction.getContextSubscriptionResults(junctionID)
    # print(traci.junction.getContextSubscriptionResults(junctionID))
    # time step update
    if step % 100 ==0:
        logging.info('----------- time step: '+str(step) + '------------')
    step += 1

traci.close()

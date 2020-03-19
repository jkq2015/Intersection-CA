import os
import sys
import traci
import traci.constants as tc
import sumolib

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from src.simulationmanager import SimulationManager
from src.simlib import setUpSimulation

# recommend reading in this order:
# scenario file ----> simulation manager ---> vehicle ---> platoon ---> intersection controller

setUpSimulation("../maps/NormalIntersection_no_TLS_simple/NormalIntersection_no_TLS.sumocfg",1)
step = 0
manager = SimulationManager(False, True, False)

# set junction listener
# junction in the map: gnej15
junctionID = 'gneJ15'
traci.junction.subscribeContext(junctionID, tc.CMD_GET_VEHICLE_VARIABLE, 20, [tc.VAR_SPEED, tc.VAR_WAITING_TIME])
print(traci.junction.getContextSubscriptionResults(junctionID))

while step < 5000:
    manager.handleSimulationStep()
    traci.simulationStep()
    # monitor output
    # print(traci.junction.getContextSubscriptionResults(junctionID))
    # time step update
    step += 1

traci.close()

import os
import sys
import traci
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from src.simulationmanager import SimulationManager
from src.simlib import setUpSimulation

# recommend reading in this order:
# scenario file ----> simulation manager ---> vehicle ---> platoon ---> intersection controller

setUpSimulation("../maps/NormalIntersection_no_TLS_simple/NormalIntersection_no_TLS.sumocfg",3)
step = 0
manager = SimulationManager(True, True, False)
while step < 5000:
    manager.handleSimulationStep()
    traci.simulationStep()
    step += 1

traci.close()

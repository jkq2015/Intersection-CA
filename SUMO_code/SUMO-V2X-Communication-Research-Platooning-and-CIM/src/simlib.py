import logging
import traci
import src.BaseDefine as BaseDefine
from sumolib import checkBinary

def flatten(l):
    # A basic function to flatten a list
    return [item for sublist in l for item in sublist]

def setUpSimulation(configFile, trafficScale = 1, platoonFlag=False):
    # Check SUMO has been set up properly
    sumoBinary = checkBinary("sumo-gui")
    # sumoBinary = checkBinary("sumo")

    # Set up logger
    logging.basicConfig(format='%(asctime)s %(message)s')
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)

    # Set up output file name
    if platoonFlag:
        additionalFilesPath = '../output/additional_platoon.xml'
        tripinfoFilesPath = '../output/tripinfo_platoon.xml'
    else:
        additionalFilesPath = '../output/additional_noPlatoon.xml'
        tripinfoFilesPath = '../output/tripinfo_noPlatoon.xml'

    # Start Simulation and step through
    traci.start(
        [sumoBinary, "-c", configFile, "--step-length", str(BaseDefine.StepLength), "--collision.action", "none",
         "--start", "--additional-files", additionalFilesPath, "--duration-log.statistics", "--tripinfo-output",
         tripinfoFilesPath, "--scale", str(trafficScale)])

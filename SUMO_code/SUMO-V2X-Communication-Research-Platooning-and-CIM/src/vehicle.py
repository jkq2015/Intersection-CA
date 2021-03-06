import traci
import src.BaseDefine as BaseDefine

class Vehicle():
    
    def __init__(self, vehicle):
        self._active = True
        self._acceleration = traci.vehicle.getAcceleration(vehicle)
        self._length = traci.vehicle.getLength(vehicle)
        self._maxSpeed = traci.vehicle.getMaxSpeed(vehicle)
        self._name = vehicle
        self._route = traci.vehicle.getRoute(vehicle)
        self._previouslySetValues = dict()
        self._maxDecel = traci.vehicle.getDecel(vehicle)
        self._maxAccel = traci.vehicle.getAccel(vehicle)

    def getAcceleration(self):
        # return self._acceleration
        # modify fixed value to lookup-each-call
        return traci.vehicle.getAcceleration(self.getName())

    def getMaxDeceleraion(self):
        return self._maxDecel

    def getMaxAcceleration(self):
        return self._maxAccel

    def isActive(self):
        return self._active
    
    def getEdge(self):
        return traci.vehicle.getRoadID(self.getName())

    def getLane(self):
        return traci.vehicle.getLaneID(self.getName())

    def getLaneIndex(self):
        return traci.vehicle.getLaneIndex(self.getName())

    def getLanePosition(self):
        # The position of the vehicle along the lane measured in m.
        return traci.vehicle.getLanePosition(self.getName())

    def getLanePositionFromFront(self):
        return traci.lane.getLength(self.getLane()) - self.getLanePosition()

    def getLeader(self):
        '''getLeader(string, double) -> (string, double)'''
        # Return the leading vehicle id together with the distance
        return traci.vehicle.getLeader(self.getName(), BaseDefine.PreVehicleDetectionRange)

    def getLength(self):
        return self._length

    def getMaxSpeed(self):
        return self._maxSpeed

    def getName(self):
        return self._name

    def getRemainingRoute(self):
        return self._route[traci.vehicle.getRouteIndex(self.getName()):]

    def getRoute(self):
        # getRoute(string) -> list(string)
        return self._route

    def getSpeed(self):
        return traci.vehicle.getSpeed(self.getName())

    def setColor(self, color):
        self._setAttr("setColor", color)

    def setInActive(self):
        self._active = False

    def setImperfection(self, imperfection):
        self._setAttr("setImperfection", imperfection)

    def setMinGap(self, minGap):
        # Sets the offset (gap to front vehicle if halting) for this vehicle.
        self._setAttr("setMinGap", minGap)

    def setTargetLane(self, lane):
        traci.vehicle.changeLane(self.getName(), lane, 0.5)

    def setTau(self, tau):
        self._setAttr("setTau", tau)

    def setSpeed(self, speed):
        # Sets the speed in m/s for the named vehicle within the last step.
        # Calling with speed=-1 hands the vehicle control back to SUMO.
        if speed == -1:
            self._setAttr("setSpeed", speed)
        else:
            lastSpeed = self.getSpeed()
            lb = lastSpeed - self._maxDecel*BaseDefine.StepLength
            ub = lastSpeed + self._maxAccel*BaseDefine.StepLength
            newSpeed = min(max(speed, lb), ub)
            self._setAttr("setSpeed", newSpeed)

    def setSpeedMode(self, speedMode):
        # bit0: Regard safe speed
        # bit1: Regard maximum acceleration
        # bit2: Regard maximum deceleration
        # bit3: Regard right of way at intersections
        # bit4: Brake hard to avoid passing a red light
        self._setAttr("setSpeedMode", speedMode)

    def setSpeedFactor(self, speedFactor):
        # speedFactor: The vehicles expected multiplicator for lane speed limits
        self._setAttr("setSpeedFactor", speedFactor)

    def _setAttr(self, attr, arg):
        # Only set an attribute if the value is different from the previous value set
        # This improves performance
        if self.isActive():
            if attr in self._previouslySetValues:
                if self._previouslySetValues[attr] == arg:
                    return
            self._previouslySetValues[attr] = arg
            getattr(traci.vehicle, attr)(self.getName(), arg)
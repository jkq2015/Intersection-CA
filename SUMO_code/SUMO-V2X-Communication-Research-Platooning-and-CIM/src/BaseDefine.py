# 对于基础变量的约定

##### 仿真相关参数
MaxStep = 10000
StepLength = 0.1    # 仿真step对应的现实时间


##### 车队相关
PlatoonMaxVehicleNum = 5    # platoon中最大车辆数
PlatoonResDistance = 10        # platoon车辆之间的期望距离


##### 车辆相关参数
PreVehicleDetectionRange = 20   # 获取前车时的检测范围（超过该范围不认为是前车）
SafeDistance = PlatoonResDistance




##### 交叉口相关
JunctionDetectionRange = 40 # 交叉口context retrieval的范围
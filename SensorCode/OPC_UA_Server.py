# OPC UA Server for Raspberry Pi
# to isntall ocua:
#       pip install freeopcua
#       pip install cryptography

from opcua import Server

import numpy as np
import mpu6050 as mp
import time

RollingSize = 2000

server = Server()
sensor = mp.mpu6050(0x68)

url = "opc.tcp://192.168.33.123:4840"
server.set_endppoint(url)

name = "Accel_Pi_Data"
addspace = server.register_namespace(name)

node = server.get_objects_node()

Param = node.add_object(addspace, "Parameters")

xAccel = Param.add_Variable(addspace, "xAccel",0)
yAccel = Param.add_Variable(addspace, "yAccel",0)
zAccel = Param.add_Variable(addspace, "zAccel",0)

xRolling = np.zeros(RollingSize)
yRolling = np.zeros(RollingSize)
zRolling = np.zeros(RollingSize)
TimeRolling = np.zeros(RollingSize)

xAccelTrend = Param.add_Variable(addspace, "xAccelTrend",xRolling)
yAccelTrend = Param.add_Variable(addspace, "yAccelTrend",yRolling)
zAccelTrend = Param.add_Variable(addspace, "zAccelTrend",zRolling)
TimeTrend = Param.add_Variable(addspace, "TimeTrend",TimeRolling)

xAccel.set_writeable()
yAccel.set_writeable()
zAccel.set_writeable()
xAccelTrend.set_writeable()
yAccelTrend.set_writeable()
zAccelTrend.set_writeable()

server.start()

while True:
    gx, gy, gz = sensor.get_accel_data(True)

    xAccel.set_value(gx)
    yAccel.set_value(gy)
    zAccel.set_value(gz)

    xRolling = np.roll(xRolling,-1)
    xRolling[-1] = gx
    xAccelTrend.set_value(xRolling)

    yRolling = np.roll(yRolling,-1)
    yRolling[-1] = gy
    yAccelTrend.set_value(yRolling)

    zRolling = np.roll(zRolling,-1)
    zRolling[-1] = gz
    zAccelTrend.set_value(zRolling)

    TimeRolling = np.roll(TimeRolling,-1)
    TimeRolling[-1] = time.time()
    TimeTrend.set_value(TimeRolling)

    
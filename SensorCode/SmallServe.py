# OPC UA Server for Raspberry Pi
# to isntall ocua:
#       pip install freeopcua
#       pip install cryptography

# Standard Python Libraries
import numpy as np
import time

# OPC Server libraries
from opcua import Server
from opcua import ua

# Accelerometer library, from .py class file in folder
import mpu6050 as mp

# Set how often you would like the data refreshed, can be 0.1 to infinity
RefreshSeconds = 5.0

server = Server()
sensor = mp.mpu6050(0x68)

# Set up the OPC server
url = "opc.tcp://192.168.0.158:4840"
server.set_endpoint(url)

uri = "http://examples.freeopuca.github.io"
name = "Accel_Pi_Data"
addspace = server.register_namespace(uri)
server.set_server_name(name)

node = server.get_objects_node()

Param = node.add_object(addspace, "Parameters")

#Add the initial 4 variables and initially set the value to 0
#      VerticalStdDev is noise level fromt he average of the signal.  could do better with Std Dev from cleaned signal,>#      VerticalMax is the maximum level from 0 that was seen in the time window
#      Horizontal readings are in perpandicular the to track and expected direction of motion
#      CurrentDataRate is the amount of Hz rate of the last data refresh cycle
VerticalStdDev = Param.add_variable(addspace, "Vertical Std Dev",0.0)
VerticalMax = Param.add_variable(addspace, "Vertical Max Value",0.0)
HorizontalStdDev = Param.add_variable(addspace, "Horizontal Std Dev",0.0)
HorizontalMax = Param.add_variable(addspace, "Horizontal Max Value",0.0)
CurrentDataRate = Param.add_variable(addspace, "Current Data Rate",300.0)

VerticalStdDev.set_writable()
VerticalMax.set_writable()
HorizontalStdDev.set_writable()
HorizontalMax.set_writable()
CurrentDataRate.set_writable()

# initialize arrays that will hold the recordings
VerticalData = []
HorizontalData =[]

#start server
server.start()

#Calibrate the sensor.  The MEMS sensors drift over time, need to callibrate occasionally
#  Make sure it is a rest, made to be mounted with the USB ports down, power and HDMI perpandicular to tracks
for j in range(5):
    Results = []
    for i in range(500):
        gx, gy, gz = sensor.get_accel_data(True)
        Results.append([time.time(), gx, gy, gz])

    Results = np.matrix(Results)

    sensor.X_OFFSET+= np.average(Results[:,1])
    sensor.Y_OFFSET+= np.average(Results[:,2])
    sensor.Z_OFFSET+= np.average(Results[:,3])

    print(sensor.X_OFFSET, sensor.Y_OFFSET, sensor.Z_OFFSET)

#initialize time
LoopStartTime = time.time()

print(' Server Started at: ', LoopStartTime)

#Loop Server forever
while True:
    #get data from sensor
    gx, gy, gz = sensor.get_accel_data(True)

    #add data to arrays
    VerticalData.append(gx)
    HorizontalData.append(gy)

    #if the request time for update has elapsed
    if time.time() - LoopStartTime > RefreshSeconds:

        #update published values
        VerticalStdDev.set_value(np.std(VerticalData))
        VerticalMax.set_value(np.max(np.abs(VerticalData)))

        HorizontalStdDev.set_value(np.std(HorizontalData))
        HorizontalMax.set_value(np.max(np.abs(HorizontalData)))
        CurrentDataRate.set_value(len(VerticalData) / RefreshSeconds)

        # re-initialize varriables
        VerticalData = []
        HorizontalData =[]

        LoopStartTime = time.time()

        
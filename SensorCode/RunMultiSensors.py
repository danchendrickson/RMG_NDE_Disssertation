#MPU-Graph
import numpy as np
import matplotlib.pyplot as plt
import multiprocessing
import time
import mpu6050mp as mp


def Calibrate(sensor):
    for j in range(10):
        Results = []
        for i in range(1000):
            gx, gy, gz = sensor.get_accel_data(True)
            Results.append([gx, gy, gz])

        Results = np.matrix(Results)

        sensor.X_OFFSET+= np.average(Results[:,0])
        sensor.Y_OFFSET+= np.average(Results[:,1])
        sensor.Z_OFFSET+= np.average(Results[:,2])
    
    return sensor
       
 
    
Sensors = []

Sensors.append(mp.mpu6050(0x68))
Sensors.append(mp.mpu6050(0x69))
Sensors.append(mp.mpu6050(0x68,3))
Sensors.append(mp.mpu6050(0x68,4))

for i in range(len(Sensors)):
    Calibrate(Sensors[i])

Result = []

f = open(time.strftime('%y%m%d-%H%M%S')+'.csv','w')

for i in range(6000):
    UnitResults = []
    AllAccels=[]
    UnitResults.append(time.time())

    line = time.strftime('%y%m%d-%H%M%S.')+str(int((time.time()-int(time.time()))*10000%10000)).zfill(4)

    for j in range(len(Sensors)):
        Accels=[]
        gx, gy, gz = Sensors[j].get_accel_data(True)
        UnitResults.append(gx)
        AllAccels.append([gx,gy,gz])
        line += ', ' + str(gx) + ', ' + str(gy)+ ', ' + str(gz)
    
    Result.append(UnitResults)

    line += '\n'
    f.write(line)

f.close()

Result = np.matrix(Result)

delTime = []
for i in range(5998):
    delTime.append(Result[i+1,0]-Result[i,0])

print(np.average(delTime))

fig = plt.figure()
plt.plot(Result[:,0],Result[:,1],label="g's x1")
plt.plot(Result[:,0],Result[:,2],label="g's x2")
plt.plot(Result[:,0],Result[:,3],label="g's x3")
plt.plot(Result[:,0],Result[:,4],label="g's x4")
plt.show()

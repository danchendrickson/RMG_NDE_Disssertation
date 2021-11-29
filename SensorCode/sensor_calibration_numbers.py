#simple Cal
import mpu6050 as mp
import time
import numpy as np

sensor = mp.mpu6050(0x68)

print('Round number: ',0)
print(sensor.X_OFFSET)
print(sensor.Y_OFFSET)
print(sensor.Z_OFFSET)
print(' ')

for j in range(10):
    Results = []
    for i in range(1000):
        gx, gy, gz = sensor.get_accel_data(True)
        Results.append([time.time(), gx, gy, gz])

    Results = np.matrix(Results)

    sensor.X_OFFSET+= np.average(Results[:,1])
    sensor.Y_OFFSET+= np.average(Results[:,2])
    sensor.Z_OFFSET+= (1+np.average(Results[:,3]))
    
    print('Round number: ',j+1)
    print(sensor.X_OFFSET, np.average(Results[:,1]))
    print(sensor.Y_OFFSET, np.average(Results[:,2]))
    print(sensor.Z_OFFSET, (1+np.average(Results[:,3])))
    print(' ')
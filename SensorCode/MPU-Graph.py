#MPU-Graph
import numpy as np
import matplotlib.pyplot as plt

import mpu6050 as mp
import time

sensor = mp.mpu6050(0x68)
Results = []

for i in range(1250):
    gx, gy, gz = sensor.get_accel_data(True)
    tx, ty, tz = sensor.get_gyro_data()
    Results.append([time.time(), gx, gy, gz, tx, ty, tz])
    #time.sleep(0.01)

Results = np.matrix(Results)

fig = plt.figure()
plt.plot(Results[:,0],Results[:,1],label="g's x")
plt.plot(Results[:,0],Results[:,2],label="g's y")
plt.plot(Results[:,0],Results[:,3],label="g's z")
title = 'Accelerometer Test'

plt.show()


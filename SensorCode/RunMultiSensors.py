#MPU-Graph
import numpy as np
import matplotlib.pyplot as plt
import multiprocessing
import time
import os
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
Sensors.append(mp.mpu6050(0x69,3))
Sensors.append(mp.mpu6050(0x68,4))
Sensors.append(mp.mpu6050(0x69,4))

for i in range(len(Sensors)):
    Calibrate(Sensors[i])

#Result = []
print('Calibrated')

if len(os.listdir('/media/pi')) ==0:
    path = '/home/pi/AccelData'
else:
    if not os.path.isdir('/media/pi/' + os.listdir('/media/pi')[0] +'/AccelData'):
        os.mkdir('/media/pi/' + os.listdir('/media/pi')[0] +'/AccelData')
    path = '/media/pi/' + os.listdir('/media/pi')[0] +'/AccelData/'


f = open(path+'/multi.' + time.strftime('%y%m%d-%H%M%S')+'.csv','w')
line = 'Date,Hour,Minute,Second,Sec Fraction'
for i in range(len(Sensors)):
    line += ',Sen'+str(i)+'x,Sen'+str(i)+'y,Sen'+str(i)+',Sen'+str(i)+'z'
line+='\n'
f.write(line)

CurrDate = time.strftime('%y%m%d')

try:
    while True:
            
        #UnitResults = []
        #AllAccels=[]
        #UnitResults.append(time.time())
        
        if CurrDate != time.strftime('%y%m%d'):
            f.close()
            f = open(path+'/multi.' + time.strftime('%y%m%d-%H%M%S')+'.csv','w')
            line = 'Date,Hour,Minute,Second,Sec Fraction'
            for i in range(len(Sensors)):
                line += ',Sen'+str(i)+'x,Sen'+str(i)+'y,Sen'+str(i)+',Sen'+str(i)+'z'
            line+='\n'
            f.write(line)
            CurrDate = time.strftime('%y%m%d')
        
        line = time.strftime('%y%m%d')+', '+time.strftime('%H')+', '+time.strftime('%M')+', '+time.strftime('%S')+', '+str(int((time.time()-int(time.time()))*10000%10000)).zfill(4)

        for j in range(len(Sensors)):
            #Accels=[]
            gx, gy, gz = Sensors[j].get_accel_data(True)
            #UnitResults.append(gx)
            #AllAccels.append([gx,gy,gz])
            line += ', ' + str(gx) + ', ' + str(gy)+ ', ' + str(gz)
        
        #Result.append(UnitResults)

        line += '\n'
        f.write(line)
except KeyboardInterrupt:
    pass

f.close()

#Result = np.matrix(Result)

#delTime = []
#for i in range(5998):
#    delTime.append(Result[i+1,0]-Result[i,0])

#print(np.average(delTime))

#fig = plt.figure()
#plt.plot(Result[:,0],Result[:,1],label="g's x1")
#plt.plot(Result[:,0],Result[:,2],label="g's x2")
#plt.plot(Result[:,0],Result[:,3],label="g's x3")
#plt.plot(Result[:,0],Result[:,4],label="g's x4")
#plt.plot(Result[:,0],Result[:,5],label="g's x5")
#plt.plot(Result[:,0],Result[:,6],label="g's x6")
#plt.show()

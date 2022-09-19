#This is for Sensor Number
SeNum = 4

#to create matrix
import numpy as np

#Sensor driver, is a py file in the same directory
import mpu6050 as mp

#for time stamps
import time

#to delete csv after created
import os

path = '/'

#set up sensor using the class from the driver
sensor = mp.mpu6050(0x68)

for j in range(10):
    Results = []
    for i in range(1000):
        gx, gy, gz = sensor.get_accel_data(True)
        Results.append([time.time(), gx, gy, gz])

    Results = np.matrix(Results)

    sensor.X_OFFSET -= np.average(Results[:,1])
    sensor.Y_OFFSET -= np.average(Results[:,2])
    sensor.Z_OFFSET -= (1+np.average(Results[:,3]))

print('Calibrated')

start = time.time()

GB = 2

#determine where to save the file, if there is a usb drive, save to usb drive, otherwise home folder.  If there is no AccelData folder, create it
if len(os.listdir('/media/pi')) == 0:
    path = '/home/pi/AccelData'
elif len(os.listdir('/media/pi')) == 1:
    if not os.path.isdir('/media/pi/' + os.listdir('/media/pi')[0] +'/AccelData'):
        os.mkdir('/media/pi/' + os.listdir('/media/pi')[0] +'/AccelData')
    path = '/media/pi/' + os.listdir('/media/pi')[0] +'/AccelData/'
else:
    for folder in os.listdir('/media/pi'):
        if folder[4:5] == '-':
            if not os.path.isdir('/media/pi/' + folder +'/AccelData'):
                os.mkdir('/media/pi/' + folder +'/AccelData')
            path = '/media/pi/' + folder +'/AccelData/'
            

#Run forever or until the drive has less than 1GB of storage space left
while GB > 1:
    #clear previous points
    Results = []
    FileStart = time.time()

    Day = time.strftime('%y%m%d')
    
    writeFile = open(path + Day + ' recording'+str(SeNum)+'.csv','a')

    #collect 60,000 data points
    while Day == time.strftime('%y%m%d'):
        try:
            gx, gy, gz = sensor.get_accel_data(True)
        except:
            del sensor
            time.sleep(0.01)
            sensor = mp.mpu6050(0x68)
            gx, gy, gz = sensor.get_accel_data(True)
        
        writeFile.write(time.strftime('%y%m%d-%H%M%S.')+str(int((time.time()-int(time.time()))*10000%10000)).zfill(4)+', '+ str(gx)+', '+ str(gy)+', '+ str(gz)+', '+ str(SeNum)+'\n')

    writeFile.close()

    #Check storage left on the drive
    st = os.statvfs(path)
    bytes_avail = (st.f_bavail * st.f_frsize)
    GB = bytes_avail / 1024 / 1024 / 1024
    

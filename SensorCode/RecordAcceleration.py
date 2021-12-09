#This is for Sensor Number
SeNum = 4

#to create matrix
import numpy as np

#Sensor driver
import mpu6050 as mp

#for time stamps
import time

#zip csv to condense size
import zipfile

#to create csv file from numpy matrix
import pandas as pd

#to delete csv after created
import os

path = '/'

#set up sensor using the class from the driver
sensor = mp.mpu6050(0x68)

start = time.time()

GB = 2

#determine where to save the file, if there is a usb drive, save to usb drive, otherwise home folder.  If there is no AccelData folder, create it
if len(os.listdir('/media/pi')) ==0:
    path = '/home/pi/AccelData'
else:
    if not os.path.isdir('/media/pi/' + os.listdir('/media/pi')[0] +'/AccelData'):
        os.mkdir('/media/pi/' + os.listdir('/media/pi')[0] +'/AccelData')
    path = '/media/pi/' + os.listdir('/media/pi')[0] +'/AccelData/'

#Run forever or until the drive has less than 1GB of storage space left
while GB > 1:
    #clear previous points
    Results = []
    FileStart = time.time()

    #collect 60,000 data points
    for i in range(60000):
        try:
            gx, gy, gz = sensor.get_accel_data(True)
        except:
            del sensor
            time.sleep(0.01)
            sensor = mp.mpu6050(0x68)
            gx, gy, gz = sensor.get_accel_data(True)
        
        Results.append([time.strftime('%y%m%d-%H%M%S.')+str(int((time.time()-int(time.time()))*10000%10000)).zfill(4), gx, gy, gz,SeNum])
        time.sleep(0.00001)

    Results = np.matrix(Results)

    #Save results to a csv file
    NextFileName = '60kPoints-'+time.strftime('%y%m%d-%H%M')+'-s'+str(SeNum)
    df = pd.DataFrame(data=Results)
    df.to_csv(NextFileName+'.csv', sep=',', header=False, float_format='%.5f')
    
    #zip the csv file
    zip_file = zipfile.ZipFile(path + NextFileName+'.zip', 'w')
    zip_file.write(NextFileName+'.csv', compress_type=zipfile.ZIP_DEFLATED)
    zip_file.close()
    
    #Delete the csv
    os.remove(NextFileName+'.csv')
    
    #Check storage left on the drive
    st = os.statvfs(path)
    bytes_avail = (st.f_bavail * st.f_frsize)
    GB = bytes_avail / 1024 / 1024 / 1024
    

#This is for Sensor Number

SeNum = 4

import numpy as np
#import matplotlib.pyplot as plt

import mpu6050 as mp
import time
import zipfile
import pandas as pd
import os
path = '/'

sensor = mp.mpu6050(0x68)

start = time.time()

j = 1

GB = 2

if len(os.listdir('/media/pi')) ==0:
    path = '/home/pi/AccelData'
else:
    if not os.path.isdir('/media/pi/' + os.listdir('/media/pi')[0] +'/AccelData'):
        os.mkdir('/media/pi/' + os.listdir('/media/pi')[0] +'/AccelData')
    path = '/media/pi/' + os.listdir('/media/pi')[0] +'/AccelData/'

while GB > 1:
    Results = []
    FileStart = time.time()
    for i in range(60000):
        try:
            gx, gy, gz = sensor.get_accel_data(True)
        except:
            del sensor
            time.sleep(0.01)
            sensor = mp.mpu6050(0x68)
            gx, gy, gz = sensor.get_accel_data(True)
        #tx, ty, tz = sensor.get_gyro_data()
        Results.append([time.strftime('%y%m%d-%H%M%S.')+str(int((time.time()-int(time.time()))*10000%10000)).zfill(4), gx, gy, gz,SeNum])
        time.sleep(0.00001)

    Results = np.matrix(Results)

    NextFileName = '60kPoints-'+time.strftime('%y%m%d-%H%M')+'-s'+str(SeNum)
    
    df = pd.DataFrame(data=Results)
    df.to_csv(NextFileName+'.csv', sep=',', header=False, float_format='%.5f')
    
    print(j, (time.time()-start) / 60.0, NextFileName)
    j+=1

    zip_file = zipfile.ZipFile(path + NextFileName+'.zip', 'w')
    zip_file.write(NextFileName+'.csv', compress_type=zipfile.ZIP_DEFLATED)
    zip_file.close()
    
    os.remove(NextFileName+'.csv')
    
    st = os.statvfs(path)
    bytes_avail = (st.f_bavail * st.f_frsize)
    GB = bytes_avail / 1024 / 1024 / 1024
    

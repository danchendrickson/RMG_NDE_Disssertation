import os
import time

if len(os.listdir('/media/pi')) ==0:
    path = '/home/pi/AccelData'
else:
    if not os.path.isdir('/media/pi/' + os.listdir('/media/pi')[0] +'/AccelData'):
        os.mkdir('/media/pi/' + os.listdir('/media/pi')[0] +'/AccelData')
    path = '/media/pi/' + os.listdir('/media/pi')[0] +'/AccelData/'

f = open(path+'/Events.' + time.strftime('%y%m%d-%H%M%S')+'.csv','w')

line = 'Day, hour, minute, second, event \n'
f.write(line)

try:
    while True:
        print('Any Event?  ')
        Comment = input()
        line = time.strftime('%y%m%d')+', '+time.strftime('%H')+', '+time.strftime('%M')+', '+time.strftime('%S')+', "'+Comment+'"\n'
        f.write(line)   

except:
    pass

f.close()
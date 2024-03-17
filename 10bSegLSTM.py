# %% [markdown]
# # Look at accelerometer data 
# 
# Finding Zero velocity times by rail axis acceleration noise levels, making summary statistics for the noise levels across the whole day files.  Spot check graphs to see what works

# %%
#Standard Header used on the projects

#first the major packages used for math and graphing
import numpy as np
import matplotlib.pyplot as plt
from cycler import cycler
import scipy.special as sp
import pickle

#Custome graph format style sheet
#plt.style.use('Prospectus.mplstyle')

#If being run by a seperate file, use the seperate file's graph format and saving paramaeters
#otherwise set what is needed
if not 'Saving' in locals():
    Saving = False
if not 'Titles' in locals():
    Titles = True
if not 'Ledgends' in locals():
    Ledgends = True
if not 'FFormat' in locals():
    FFormat = '.png'

#Standard cycle to make black and white images and dashed and line styles
default_cycler = (cycler('color', ['0.00', '0.40', '0.60', '0.70']) + cycler(linestyle=['-', '-', '-', '-']))
plt.rc('axes', prop_cycle=default_cycler)
my_cmap = plt.get_cmap('gray')

# %%
#Extra Headers:
import os as os
import pywt as py
import statistics as st
import os as os
import random
import multiprocessing
from joblib import Parallel, delayed
import platform

from time import time as ti

# %%
import CoreFunctions as cf
from skimage.restoration import denoise_wavelet
# %%
from sklearn.model_selection import train_test_split
import dask.dataframe as dd


# %%
from keras.layers import LSTM, Dense, RepeatVector, TimeDistributed, Masking
from keras.callbacks import ModelCheckpoint
from keras.models import Sequential
import tensorflow as tf

# %% [markdown]
# ## Choosing Platform
# Working is beinging conducted on several computers, and author needs to be able to run code on all without rewriting..  This segment of determines which computer is being used, and sets the directories accordingly.

# %%
HostName = platform.node()

if HostName == "Server":
    Computer = "Desktop"   
elif HostName[-6:] == 'wm.edu':
    Computer = "SciClone"
elif HostName == "SchoolLaptop":
    Computer = "LinLap"
elif HostName == "WTC-TAB-512":
    Computer = "PortLap"
else:
    Computer = "WinLap"

# %%
if Computer == "SciClone":
    location = '/sciclone/home/dchendrickson01/image/'
elif Computer == "WinLap":
    location = 'C:\\Data\\'
elif Computer == "Desktop":
    location = "E:\\Backups\\Dan\\CraneData\\"
elif Computer == "LinLap":
    location = '/home/dan/Output/'
elif Computer == 'PortLap':
    location = 'C:\\users\\dhendrickson\\Desktop\\AccelData\\'

# %%
if Computer ==  "SciClone":
    rootfolder = '/sciclone/home/dchendrickson01/'
    folder = '/scratch/Recordings2/'
    imageFolder = '/scratch/Move3Dprint/'
elif Computer == "Desktop":
    rootfolder = location
    folder = rootfolder + "Recordings2\\"
elif Computer =="WinLap":
    rootfolder = location
    folder = rootfolder + "Recordings2\\"   
elif Computer == "LinLap":
    rootfolder = '/home/dan/Data/'
    folder = rootfolder + 'Recordings2/'
elif Computer =='PortLap':
    rootfolder = location 
    folder = rootfolder + 'Recordings2\\'

# %% [markdown]
# ## Global Variables

# %%
Saving = False
location = folder
Titles = True
Ledgends = True

f = 0


# %%
files = ['230418 recording1.csv','230419 recording1.csv','230420 recording1.csv','230421 recording1.csv',
         '230418 recording2.csv','230419 recording2.csv','230420 recording2.csv','230421 recording2.csv',
         '230425 recording1.csv','230425 recording2.csv','230426 recording2.csv','230427 recording2.csv',
         '230428 recording2.csv','230509 recording1.csv','230510 recording1.csv','230511 recording1.csv']

# %%
BeforeTamping = ['221206 recording1.csv','221207 recording1.csv','221208 recording1.csv','221209 recording1.csv',
         '221206 recording2.csv','221207 recording2.csv','221208 recording2.csv','221209 recording2.csv']


# %% [markdown]
# ## Project Specific Functions

# %%
def RollingStdDev(RawData, SmoothData, RollSize = 25):
    StdDevs = []
    for i in range(RollSize):
        Diffs = RawData[0:i+1]-SmoothData[0:i+1]
        Sqs = Diffs * Diffs
        Var = sum(Sqs) / (i+1)
        StdDev = np.sqrt(Var)
        StdDevs.append(StdDev)
    for i in range(len(RawData)-RollSize-1):
        j = i + RollSize
        Diffs = RawData[i:j]-SmoothData[i:j]
        Sqs = Diffs * Diffs
        Var = sum(Sqs) / RollSize
        StdDev = np.sqrt(Var)
        StdDevs.append(StdDev)  
    
    return StdDevs

def RollingSum(Data, Length = 100):
    RollSumStdDev = []
    for i in range(Length):
        RollSumStdDev.append(sum(Data[0:i+1]))
    for i in range(len(Data) - Length):
        RollSumStdDev.append(sum(Data[i:i+Length]))
    return RollSumStdDev

def SquelchPattern(DataSet, StallRange = 5000, SquelchLevel = 0.02):
    SquelchSignal = np.ones(len(DataSet))

    for i in range(len(DataSet)-2*StallRange):
        if np.average(DataSet[i:i+StallRange]) < SquelchLevel:
            SquelchSignal[i+StallRange]=0

    return SquelchSignal

def getVelocity(Acceleration, Timestamps = 0.003, Squelch = [], corrected = 0):
    velocity = np.zeros(len(Acceleration))
    
    Acceleration -= np.average(Acceleration)
    
    if len(Timestamps) == 1:
        dTime = np.ones(len(Acceleration),dtype=float) * Timestamps
    elif len(Timestamps) == len(Acceleration):
        dTime = np.zeros(len(Timestamps), dtype=float)
        dTime[0]=1
        for i in range(len(Timestamps)-1):
            j = i+1
            if Timestamps[j] > Timestamps[i]:
                dTime[j]=Timestamps[j]-Timestamps[i]
            else:
                dTime[j]=Timestamps[j]-Timestamps[i]+10000.0
        dTime /= 10000.0

    velocity[0] = Acceleration[0] * (dTime[0])

    for i in range(len(Acceleration)-1):
        j = i + 1
        if corrected ==2:
            if Squelch[j]==0:
                velocity[j]=0
            else:
                velocity[j] = velocity[i] + Acceleration[j] * dTime[j]                
        else:
            velocity[j] = velocity[i] + Acceleration[j] * dTime[j]

    if corrected == 1:
        PointVairance = velocity[-1:] / len(velocity)
        for i in range(len(velocity)):
            velocity[i] -=  PointVairance * i
    
    velocity *= 9.81

    return velocity

def MakeDTs(Seconds, Miliseconds):
    dts = np.zeros(len(Miliseconds), dtype=float)
    dts[0]=1
    for i in range(len(MiliSeconds)-1):
        j = i+1
        if Seconds[j]==Seconds[i]:
            dts[j]=Miliseconds[j]-Miliseconds[i]
        else:
            dts[j]=Miliseconds[j]-Miliseconds[i]+1000
    dts /= 10000
    return dts



# %%
#Smooth = cf.Smoothing(ODataSet[:,3],2) #,50)
def DeviationVelocity(file):
    if file[-3:] =='csv':
        ODataSet = np.genfromtxt(open(folder+file,'r'), delimiter=',',skip_header=0,missing_values=0,invalid_raise=False)
        SmoothX = denoise_wavelet(ODataSet[:,3], method='VisuShrink', mode='soft', wavelet_levels=3, wavelet='sym2', rescale_sigma='True')
        SmoothY = denoise_wavelet(ODataSet[:,4], method='VisuShrink', mode='soft', wavelet_levels=3, wavelet='sym2', rescale_sigma='True')
        SmoothZ = denoise_wavelet(ODataSet[:,5], method='VisuShrink', mode='soft', wavelet_levels=3, wavelet='sym2', rescale_sigma='True')
        SmoothX -= np.average(SmoothX)
        SmoothY -= np.average(SmoothY)
        SmoothZ -= np.average(SmoothZ)
        StdDevsX = RollingStdDev(ODataSet[:,3],SmoothX)
        StdDevsX.append(0)
        StdDevsX = np.asarray(StdDevsX)
        SmoothDevX = denoise_wavelet(StdDevsX, method='VisuShrink', mode='soft', wavelet_levels=3, wavelet='sym2', rescale_sigma='True')
        SquelchSignal = SquelchPattern(SmoothDevX, 2000, 0.03)
        #Velocity = getVelocity(ODataSet[:,3], ODataSet[:,2],SquelchSignal, 2)
        #Velocity = np.asarray(Velocity)
        MoveMatrix = np.matrix([SmoothX, SmoothY, SmoothZ])
        return [SquelchSignal,MoveMatrix,SmoothDevX,file[:-4]]
    else:
        pass


# %%
def SepreateMovements(SquelchSignal, RawData, FileName):
    Moves= []
    MoveNames = []
    Move = np.zeros((1,3), dtype=float)
    i = 0
    for j in range(len(SquelchSignal)-1):
        if SquelchSignal[j] == 1:
            #try:
            Move = np.concatenate((Move, RawData[j,:]), axis=0)
            #except:
            #    print(j)
            if SquelchSignal[j+1] == 0:
                #Move = np.matrix(Move)
                Moves.append(Move)
                MoveNames.append(FileName + str(i).zfill(3))
                i+=1
                Move = np.zeros((1,3), dtype=float)
                #Move[0,2]=0
    Moves.append(Move)
    MoveNames.append(FileName + str(i).zfill(3))
    return (Moves, MoveNames)
    


# %%
def splitLong(Moves, maxLength = 4000, minLength = 1000, MoveNames = []):
    if len(MoveNames) <=1:
        MoveNames = ['null'  for x in range(len(Moves))]
    Xmoves = []
    Xnames = []
    for i in range(len(Moves)):
        if np.shape(Moves[i])[0] > maxLength: 
            Xmoves.append(Moves[i][:int(len(Moves[i])/2),:])
            Xnames.append(MoveNames[i] + 'a')
            Xmoves.append(Moves[i][int(len(Moves[i])/2):,:])
            Xnames.append(MoveNames[i] + 'b')
        else:
            if np.shape(Moves[i])[0] < minLength:
                pass
            else:
                Xmoves.append(Moves[i])
                Xnames.append(MoveNames[i])
    return Xmoves, Xnames

def findMaxLength(Moves):
    maxLength = 0
    LongMove = 0
    for i in range(len(Moves)):
        if np.shape(Moves[i])[0] > maxLength: 
            maxLength =  np.shape(Moves[i])[0]
            LongMove = i
    return maxLength, LongMove

def findMinLength(Moves):
    minLength = 9999999
    SmallMove = 0
    for i in range(len(Moves)):
        if np.shape(Moves[i])[0] < minLength: 
            minLength =  np.shape(Moves[i])[0]
            SmallMove = i
    return minLength, SmallMove



# %% [markdown]
# ## Process Files

# %%
LoopFiles = 16
loops = int(len(files) / LoopFiles) 
if len(files)%LoopFiles != 0:
    loops += 1

print('files', len(files), loops)

# %%
SquelchSignal = []
RawData=[]
OrderedFileNames=[]


# %%

st = ti()

for k in range(loops):
    if k == loops -1:
        tfiles = files[k*LoopFiles:]
    else:
        tfiles = files[k*LoopFiles:(k+1)*LoopFiles]
    Results = Parallel(n_jobs=LoopFiles)(delayed(DeviationVelocity)(file) for file in tfiles)
    #Results =[]
    #for file in tfiles:
    #    Results.append(DeviationVelocity(file))
    #    print(file, (ti()-st)/60.0)
        
    for i in range(len(Results)):       
        SquelchSignal.append(Results[i][0])
        RawData.append(Results[i][1])
        OrderedFileNames.append(Results[i][3])
    print(k, len(Results), (ti()-st)/60.0)
    



# %%
print('got data', len(RawData), len(SquelchSignal), np.shape(RawData[0]))

# %%
#MoveData = Parallel(n_jobs=31)(delayed(SepreateMovements)(SquelchSignal[i], RawData[i], OrderedFileNames[i])
#                                       for i in range(len(RawData)))

MoveData = []
Moves = []
MoveNames = []

for i in range(len(RawData)):
    temp = SepreateMovements(SquelchSignal[i], RawData[i].T, OrderedFileNames[i])
    print(i, len(temp), len(temp[0]), len(temp[1]))
    for j in range(len(temp[0])):
        Moves.append(temp[0][j])
        MoveNames.append(temp[1][j])
    #MoveData.append(temp)

print('Moves Seperated', len(Moves), (ti()-st)/60.0)
    
# %%
#Movements = []
#GroupNames = []
#for move in MoveData:
#    Movements.append(move[0])
#    GroupNames.append(move[1])

#print('Move and Name Sepreated', len(Movements), (ti()-st)/60.0)

# %%
#Moves=[]

#for Groups in Movements:
#    for Move in Groups:
#        Moves.append(np.asarray(Move).astype('float32'))
#Moves = np.asarray(Moves)

#MoveNames = []
#for Groups in GroupNames:
#    for name in Groups:
#        MoveNames.append(name)

# %%
#print('made moves')

del SquelchSignal
del RawData
#del Movements
#del GroupNames
#del MoveData
#del OrderedFileNames


# %%
longMove, MoveNumb = findMaxLength(Moves)


# %%
minLength = 750

# %%
Moves, MoveNames = splitLong(Moves, longMove+1, minLength, MoveNames)

print('Post split length', len(Moves), np.shape(Moves[0]))

# %%
#padding moves.  Not needed, need sequences, not moves

#Moves2 = []
#for move in Moves:
#    if np.shape(move)[0] < longMove:
#        padding = np.zeros((longMove-np.shape(move)[0], 3))
#        tempMove = np.concatenate((move, padding), axis=0)
#        Moves2.append(tempMove)
#    else:
#        Moves2.append(move)
#Moves = Moves2

#del Moves2

# %% [markdown]
# ## Try LSTM Stuff

# %% [markdown]
# https://machinelearningmastery.com/how-to-develop-lstm-models-for-time-series-forecasting/

# %%
TimeSteps = 250
Features = np.shape(Moves[0])[1]

# %%
# split a multivariate sequence into samples
def split_sequences(sequences, n_steps):
    X, y = list(), list()
    for i in range(len(sequences)):
        # find the end of this pattern
        end_ix = i + n_steps
        # check if we are beyond the dataset
        if end_ix > len(sequences)-1:
            break
        # gather input and output parts of the pattern
        seq_x, seq_y = sequences[i:end_ix, :], sequences[end_ix, :]
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)

# %%
Sequences = []
Outputs = []
for move in Moves:
    Seq, Out = split_sequences(move,TimeSteps)
    Sequences.append(Seq)
    Outputs.append(Out)
    

# %%
MoveSegments = []
for seq in Sequences:
    for mv in seq:
        MoveSegments.append(mv)
NextDataPoint = []
for out in Outputs:
    for pt in out:
        NextDataPoint.append(pt)

# %%
print('data split', len(MoveSegments))
print('Shape next data point', np.shape(NextDataPoint))


file=open('/scratch/PrepAccel/Data-20240307-16files-250pts.p','wb')
pickle.dump([MoveSegments, NextDataPoint],file)
file.close()


#gpus = tf.config.list_physical_devices('GPU')
#tf.config.set_visible_devices(gpus[:4], 'GPU')

tf.get_logger().setLevel('INFO')
tf.autograph.set_verbosity(1)

filePrefix = 'Run20240305-'

class LSTM_Autoencoder:
  def __init__(self, optimizer='adam', loss='mse'):
    self.optimizer = optimizer
    self.loss = loss
    self.n_features = Features
    self.timesteps = TimeSteps
    
  def build_model(self):
    timesteps = self.timesteps
    n_features = self.n_features
    tf.debugging.set_log_device_placement(True)
    gpus = tf.config.list_logical_devices('GPU')
    strategy = tf.distribute.MirroredStrategy(gpus)
    with strategy.scope():
        model = Sequential()
    
    # Padding
    #model.add(Masking(mask_value=0.0, input_shape=(timesteps, n_features)))

    # Encoder
        model.add(LSTM(timesteps, activation='relu', input_shape=(timesteps, n_features), return_sequences=True))
        model.add(LSTM(75, activation='relu', return_sequences=True))
        model.add(LSTM(8, activation='relu'))
        model.add(RepeatVector(timesteps))

        # Decoder
        model.add(LSTM(timesteps, activation='relu', return_sequences=True))
        model.add(LSTM(75, activation='relu', return_sequences=True))
        model.add(TimeDistributed(Dense(n_features)))

        model.compile(optimizer=self.optimizer, loss=self.loss)
        model.summary()
    self.model = model
    
  def fit(self, X, epochs=3, batch_size=16, callbacks = []):
    #self.timesteps = np.shape(X)[0]
    self.build_model()
    
    #input_X = np.expand_dims(X, axis=1)
    self.model.fit(X, X, epochs=epochs, batch_size=batch_size, callbacks=callbacks,verbose=0)
    
  def predict(self, X):
    #input_X = np.expand_dims(X, axis=1)
    output_X = self.model.predict(input_X)
    reconstruction = np.squeeze(output_X)
    return np.linalg.norm(X - reconstruction, axis=-1)
  
  def plot(self, scores, timeseries, threshold=0.95):
    sorted_scores = sorted(scores)
    threshold_score = sorted_scores[round(len(scores) * threshold)]
    
    plt.title("Reconstruction Error")
    plt.plot(scores)
    plt.plot([threshold_score]*len(scores), c='r')
    plt.savefig(filePrefix+'reconstructionError.png')
    plt.show()
    
    anomalous = np.where(scores > threshold_score)
    normal = np.where(scores <= threshold_score)
     
    plt.title("Anomalies")
    plt.scatter(normal, timeseries[normal][:,-1], s=3)
    plt.scatter(anomalous, timeseries[anomalous][:,-1], s=5, c='r')
    plt.savefig(filePrefix+'Anomalies.png')
    plt.show()


# %%
#split = int(len(Moves)*.9)
#Train_data = Moves[:split]
#Test_data = Moves[split:]
#Train_data = tf.ragged.constant(Train_data)
#Test_data = tf.ragged.constant(Test_data)

# %%
seq_train, seq_test, out_train, out_test = train_test_split(MoveSegments, NextDataPoint, test_size=0.05, shuffle=True, random_state=0)

# %%
#seq_train = dd.from_pandas(seq_train)

# %%
lstm_autoencoder = LSTM_Autoencoder(optimizer='adam', loss='mse')

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # Set to 2 to suppress INFO messages


# %%
print('About to start Training')

# define the checkpoint
filepath = "LSTMonMovementsmodel.h5"
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
callback_list = [checkpoint]

lstm_autoencoder.fit(seq_train, epochs=5, batch_size=32, callbacks=callback_list)

print('about to save')

lstm_autoencoder.model.save("LSTM_onMovementsmodel")



# %%
np.shape(Train_data[1])

# %%
scores = lstm_autoencoder.predict(Test_data)

# %%
lstm_autoencoder.plot(scores, Test_data, threshold=0.95)

# %%



# %%


# %%


# %%




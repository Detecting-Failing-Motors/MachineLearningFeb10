
# coding: utf-8

# In[1]:


import pickle


# In[2]:


#Import Needed Created Functions

#Need 2 Functions to extract all the data from csv files
from ExtractDataFunctions import ExtractAccelerometerData
from ExtractDataFunctions import ExtractAcousticData

#Need 3 Functions to Organize all the all the data
from OrganizationFunctions import Inputs2CondensedForm
from OrganizationFunctions import System2CondensedForm
from OrganizationFunctions import AllData2WorkingForm

#Need 1 Function to Organize Files into TestDataFrame
from FeatureFunctions import getTESTDataFrame

#Need 2 Functions for Graphing
from FeatureFunctions import getGraphs
from FeatureFunctions import getQuickPlot

#Need # Functions to perform Machine Learning
from MachineLearningFunctions import getTESTMatrix
from MachineLearningFunctions import GenerateTrainingFile
from MachineLearningFunctions import GetSplitTrainingData
from MachineLearningFunctions import GetTrainingData
from MachineLearningFunctions import TrainModel
from MachineLearningFunctions import PredictModel
from MachineLearningFunctions import PredictProbModel
from MachineLearningFunctions import GetReducedFeaturesFromDataFrame


# In[3]:


#Receive GUI Inputs

#General
Name_ID = "1"
Application = "2"
ModelNumber = "3"
SavingAlias = "4"
#AccelerometerDataFilename = 'AccelerometerActualData.csv' #filename #Origninal / Additionally File
AccelerometerDataFilename = 'AccelerometerActualDataEdited.csv' #filename #main file
#AcousticDataFilename = 'DataOutputMic.csv' #Original File #Wave File Name
AcousticDataFilename = 'DataOutputMic2Col.csv' #Wave File Name
MLDataFilename = "MLSynthesizedData.csv"

#Motor Characteristics #Need to get with Brendan
Horsepower = "6"
RatedVoltage = "7"
ACorDC = "DC"
NumberOfPolePairs = "9"
NumberofShafts = "10"

#Bearing Information
ShaftSpeed = 300 #Also Used for Motor Characteristics
NumberOfRollingElements = 3
DiameterOfRollingElements = 3
PitchDiameter = .2
ContactAngle = .2

#Processing Information 
AccelerometerSamplingFrequency = 14000 #must be an non-zero int or float
AcousticSamplingFrequency = 20000 #must be an non-zero int or float


# In[4]:


#Microcontroller Information
#Receive Required Information
A2DResolution = 16
VoltageMax = 5
VoltageMin = 0


# In[5]:


#System/Sensor Known Constants
AccelerationMax = 50 
AccelerationMin = -50


# In[6]:


#Convert User Inputs into a condensed form
OnlyUserInput = Inputs2CondensedForm(Name_ID, Application, ModelNumber, SavingAlias,                                     AccelerometerDataFilename, AcousticDataFilename,                                      MLDataFilename, Horsepower, RatedVoltage, ACorDC,                                      NumberOfPolePairs, NumberofShafts,                                      ShaftSpeed, NumberOfRollingElements,                                      DiameterOfRollingElements,PitchDiameter, ContactAngle,                                      AccelerometerSamplingFrequency, AcousticSamplingFrequency)

SystemInputs = System2CondensedForm(A2DResolution,VoltageMax,VoltageMin,AccelerationMax,AccelerationMin)


# In[7]:


#Acquire Accelerometer Actual Data
time, amp, Voltage, Acceleration = ExtractAccelerometerData(OnlyUserInput,SystemInputs)

#Acquire Acoustic Actual Data
Channel1Time,Channel1Value,Channe21Time,Channe2Value = ExtractAcousticData(OnlyUserInput,SystemInputs)


# In[8]:


#Put All Data into Working Form
trial = 2 #Select the instance of the data
AllData = AllData2WorkingForm(OnlyUserInput,SystemInputs,time[trial],                               amp[trial], Voltage[trial], Acceleration[trial],                             Channel1Time,Channel1Value,Channe21Time,Channe2Value)


# In[ ]:


#Prepare for Machine Learning
TestDF = getTESTDataFrame(AllData)
TestMatrix = getTESTMatrix(TestDF)
ShortDF = GetReducedFeaturesFromDataFrame(TestDF)
ShortTestMatrix = getTESTMatrix(ShortDF)


# In[ ]:


#Current Saved ShortModel
ShortSaveFilename = 'ShortModel.sav' #saving name
LoadedClassifier = pickle.load(open(ShortSaveFilename, 'rb')) #Load the saved model using Pickle
prediction,prediction_string = PredictModel(LoadedClassifier,ShortTestMatrix)
prediction_proba = PredictProbModel(LoadedClassifier,ShortTestMatrix)
#Output Results
print('Loaded Short Model:\n')
print(prediction_string)
print(prediction_proba)


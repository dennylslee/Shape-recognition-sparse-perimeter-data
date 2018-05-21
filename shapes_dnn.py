import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
from matplotlib import style
from mpl_toolkits.mplot3d import Axes3D

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.utils import to_categorical
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

# --------------------- Configuration ----------------------------------------
# training data control
dataset_rawsize = 1000					# dataset size per shape being recognized; in this case it is three
dataset_size =  dataset_rawsize * 2 # will split up half for testing later 
shape_points = 5
num_classes = 3							# 3 classes for square, circle and triangle
# DNN control 
num_firstlayer = 40
num_secondlayer = 20
epochs = 50
train_test_split = 0.25		# pct left for test validation during training. Keras use the latter part of dataset
batch_size = 10
# plot style control
style.use('ggplot')
plotmanifold = False

# ------------------ FORM THE RAW SQUARE TRAINING SET -----------------------------------------
# a function to generate a set of random coordinates that lies on the square perimeter
# the square is centered around zero with corners at 1 and -1
def gen_square():
	datapointset = np.random.uniform(low=-1, high=1, size=(shape_points,2))  # generate set of coordinates
	#print(datapointset)
	index = np.arange(1,shape_points+1,1)
	df = pd.DataFrame(data=datapointset, index=index, columns =['x','y'])
	df['absx'] = np.abs(df.x)
	df['absy'] = np.abs(df.y)
	df['snapaxis'] = np.where(df['absx']>df['absy'], 'x', 'y')  # decide which axis is the snapaxis
	#print(df)
	df['x']=np.where(df['snapaxis']=='x', np.sign(df.x), df.x )  # snap to -1 or +1 edge for x
	df['y']=np.where(df['snapaxis']=='y', np.sign(df.y), df.y )  # snap to -1 or +1 edge for y
	df.drop(['absx','absy','snapaxis'], axis=1, inplace=True)
	return df.values

# generate a numpy array with a set of square coordinate set
squareset = np.empty(shape = [dataset_size, shape_points + 1, 2]) # add one for label
for i in range(dataset_size):
	# extra index bracket to keep shape correct, use axis otherwise flattened
	squareset[i] = np.append(gen_square(), [[np.NaN, 0]], axis = 0)
#print(squareset[:5])
print('Rectangle training set shape:', squareset.shape)

# ------------------ FORM THE RAW CRICLE TRAINING SET -----------------------------------------
# a function to generate a set of random coordinates that lies on the circle perimeter
# the circle is centered around zero and has a radius of one
# a set of random angles in radian is first generated and converted to x, y coorindates
def gen_circle():
	datapointset = np.random.uniform(low=0, high=2*np.pi, size=(shape_points,1))  # generate set of random angles in radian
	#print(datapointset)
	index = np.arange(1,shape_points+1,1)
	df = pd.DataFrame(data=datapointset, index=index, columns =['angle_rad'])
	df['x'] = np.cos(df['angle_rad'])
	df['y'] = np.sin(df['angle_rad'])
	df.drop(['angle_rad'], axis=1, inplace=True)
	return df.values

# generate a numpy array with a set of circle coordinate set
circleset = np.empty(shape = [dataset_size, shape_points+1, 2])
for i in range(dataset_size):
	# extra index bracket to keep shape correct, use axis otherwise flattened
	circleset[i] = np.append(gen_circle(), [[np.NaN, 1]], axis = 0)
#print(circleset[:5])
print('Circle training set shape:', circleset.shape)

# ------------------ FORM THE RAW TRIANGLE TRAINING SET -----------------------------------------
# a function to generate a set of random coordinates that lies on the triangle perimeter
# the triangle is first layout as an inverse equalateral triangle pointing at (0,0)
# each side is unit 2 in length and the top edge is from  -1 to +1 on the x-axis
# the datapoints are then shift down (offset) to center the triangle at zero.
def gen_triangle():
	tri_datapt = np.empty(shape = [shape_points, 2])
	for i in range(shape_points):
		tri_edge = np.random.choice(['a','b','c'])
		if tri_edge == 'a':
			pt_x = np.random.uniform(low=-1, high=1)
			pt_coord = np.array([pt_x, math.sqrt(3)])
		elif tri_edge == 'b':
			pt_x = np.random.uniform(low=0, high=1)
			pt_coord = np.array([pt_x, math.tan(pt_x*2*math.pi/6)])  # tangent 60 deg to find the y coordinate
		else:
			pt_x = np.random.uniform(low=-1, high=0)
			pt_coord = np.array([pt_x, math.tan(abs(pt_x)*2*math.pi/6)])  # cosine 60 deg to find the y coordinate
		tri_datapt[i] =  pt_coord

	# center the triangle by moving it down on the y-axis (by subtracting root-three over two)
	tri_datapt = tri_datapt - [0, math.sqrt(3)/2]
	return tri_datapt

# generate a numpy array with a set of circle coordinate set
triangleset = np.empty(shape = [dataset_size, shape_points+1, 2])
for i in range(dataset_size):
	# extra index bracket to keep shape correct, use axis otherwise flattened
	triangleset[i] = np.append(gen_triangle(), [[np.NaN, 2]], axis = 0)
#print(triangleset[:5])
print('Triangle training set shape:', triangleset.shape)

# ------------------ CONCAT INTO ONE TRAINING SET -----------------------------------------
mainset = np.vstack((squareset, circleset, triangleset))
np.random.shuffle(mainset)						# mix up all the shape before split
# split into training and testing set
trainX = mainset[:dataset_rawsize*3, :-1]		# extract all the data points columns
trainY = mainset[:dataset_rawsize*3, -1, 1].astype(int) 		# extract last label column
testX =  mainset[dataset_rawsize*3:, :-1]		# extract all the data points columns
testY = mainset[dataset_rawsize*3:, -1, 1].astype(int)  		# extract last label column

trainY = to_categorical(trainY, num_classes = num_classes)
testY = to_categorical(testY, num_classes = num_classes)

print('Training set shape:', trainX.shape)
print('Testing set shape:', testX.shape)
print(trainX[:5])
print(trainY[:5])
print(testX[:5])
print(testY[:5])

# ------------------ BUILD DNN MODEL and TEST MODEL -----------------------------------------
# create model
model = Sequential()
model.add(Flatten(input_shape=(5,2))) #get rid of the coordinate pairing
model.add(Dense(num_firstlayer, kernel_initializer='uniform', activation='relu'))
model.add(Dense(num_secondlayer, kernel_initializer='uniform', activation='relu'))
model.add(Dense(3, kernel_initializer='uniform', activation='sigmoid'))
# Compile model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# Fit the model
model.fit(trainX, trainY, epochs = epochs, batch_size= batch_size, validation_split = train_test_split, verbose=2)
print(model.summary())
# do prediction
prediction = model.predict(testX, batch_size = batch_size, verbose = 2)
# find the max probability as predicted by DNN
predictionOneShot=np.zeros(prediction.shape, dtype = np.int)
for index, i in enumerate(prediction.argmax(axis=1)):
	predictionOneShot[index,i] = 1
print(prediction[:10])
print(predictionOneShot[:10])
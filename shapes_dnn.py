import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
from matplotlib import style
from mpl_toolkits.mplot3d import Axes3D
#import keras
#from keras.models import Sequential
#from keras.layers import Dense, Dropout
#from sklearn.metrics import confusion_matrix
#from sklearn.metrics import classification_report

# --------------------- Configuration ----------------------------------------
# training data control
dataset_rawsize = 4
dataset_size =  dataset_rawsize * 2 # will split up half for testing later 
Normalrange  = 1000
add_rand = True # switch to add in the random negative sample set
# control how spread the noise is
norm_mu,norm_sigma = 0, 10
# DNN control 
num_firstlayer = 40
num_secondlayer = 20
epochs = 50
train_test_split = 0.25		# pct left for test validation. Keras use the latter port of dataset
batch_size = 10
# plot style control
style.use('ggplot')
plotmanifold = False

shape_points = 5

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
	squareset[i] = np.append(gen_square(), [[np.NaN, 1]], axis = 0)
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
	circleset[i] = np.append(gen_circle(), [[np.NaN, 2]], axis = 0)
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
	triangleset[i] = np.append(gen_triangle(), [[np.NaN, 3]], axis = 0)
#print(triangleset[:5])
print('Triangle training set shape:', triangleset.shape)

# ------------------ CONCAT INTO ONE TRAINING SET -----------------------------------------
mainset = np.vstack((squareset, circleset, triangleset))
np.random.shuffle(mainset)						# mix up all the shape before split
# split into training and testing set
trainX = mainset[:dataset_rawsize*3, :-2]		# extract all the data points columns
trainY = mainset[:dataset_rawsize*3, -1]  		# extract last label column
testX =  mainset[dataset_rawsize*3:, :-2]		# extract all the data points columns
testY = mainset[dataset_rawsize*3:, -1]  		# extract last label column
print(trainX)
print(trainY)
print(testX)
print(testY)
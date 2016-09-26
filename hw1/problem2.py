from scipy.io import loadmat
import matplotlib.pyplot as plt 
from matplotlib import cm
import numpy
import math
import random

orc = loadmat('ocr.mat')
testlabels = orc['testlabels']
test = numpy.array(orc['testdata'], dtype = numpy.float)
# Pre-process the test data so dont have to calculate it over and over again.
# Won't shorten a single trail though, however, could save some time for 10 trails
#test_square = numpy.sum(numpy.square(test), axis = 1)
total = len(test)
train = numpy.array(orc['data'], dtype = numpy.float)
labels = orc['labels']

# Calculate the euclidean distance according to (a-b)^2 = a^2 + b^2 - 2ab
# Find the nearest neighbor for each test vector by calling numpy.argmin
# Return
def classifier(train, label, test):
	train_len = len(train)
	test_len = len(test)
	train_square = numpy.sum(numpy.square(train), axis = 1)
	test_square = numpy.sum(numpy.square(test), axis = 1)
	train_square = numpy.tile(train_square, (test_len, 1))
	testsquare = numpy.tile(test_square.transpose(), (train_len, 1)).T
	times = 2 * test.dot(train.T)
	distance = train_square + testsquare - times
	return label[numpy.argmin(distance, axis = 1)]

def prototypeSelector(train, label, m):
	train_len = len(train)
	label_counter = [0] * 10
	indeces = [0] * (m/2)

	# Offsets are used to divide the training data, in my code I divide
	offsets = [0] * 11
	counter = 0
	for x in range(0, 11):
		offsets[x] = train_len/10 * counter
		counter += 1

	index = 0
	for x in range(0, 60000):
		if (label_counter[label[x][0]] < m/20):
			indeces[index] = x
			index += 1
			label_counter[label[x][0]] += 1

	for x in range(0, 10):
		pred = classifier(train[indeces], label[indeces], train[offsets[x]:offsets[x+1]])
		difference = pred - label[offsets[x]:offsets[x+1]]
		diff = numpy.nonzero(difference)[0] + offsets[x]
		if len(diff) > m/20:
			sel = random.sample(xrange(len(diff)), m/20)
			indeces = indeces + diff[sel].tolist()
		else:
			indeces = indeces + diff.tolist()
	return {'train_set': train[indeces], 'label_set': label[indeces]}

m = [1000, 2000, 4000, 8000]
for x in range(0, 10):
	for i in m:
		result = prototypeSelector(train, labels, i)
		train_set = result['train_set']
		label_set = result['label_set']
		pred = classifier(train_set, label_set, test)
		diff = pred - testlabels
		print "Working on trial %d, m = %d, error_rate = %f" % (x + 1, i, numpy.count_nonzero(diff)/ float(total))














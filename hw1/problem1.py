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
test_square = numpy.sum(numpy.square(test), axis = 1)
total = len(test)


# Calculate the euclidean distance according to (a-b)^2 = a^2 + b^2 - 2ab
# Find the nearest neighbor for each test vector by calling numpy.argmin
# Return
def classifier(train, label, test):
	global test_square
	train_len = len(train)
	test_len = len(test)
	train_square = numpy.sum(numpy.square(train), axis = 1)
	train_square = numpy.tile(train_square, (test_len, 1))
	testsquare = numpy.tile(test_square.transpose(), (train_len, 1)).T
	times = 2 * test.dot(train.T)
	distance = train_square + testsquare - times
	return label[numpy.argmin(distance, axis = 1)]

# Run 10 trials, each trial contains 4 subtrials.
for x in range(0, 10):
	n = [1000, 2000, 4000, 8000]
	for i in n:
		sel = random.sample(xrange(60000), i)
		train = numpy.array(orc['data'][sel], dtype = numpy.float)
		labels = orc['labels'][sel]
		pred = classifier(train, labels, test)
		difference = pred-testlabels
		counter = numpy.count_nonzero(difference) # Calculate error_rate
		# Print results
		print "Working on trial %d, n = %d, error_rate = %f" % (x+1, i, counter / float(total))




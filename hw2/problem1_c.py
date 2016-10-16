from scipy.io import loadmat
import numpy as np 
import sys

# Load data and labels
news = loadmat('news.mat')
train = news['data'].toarray()
labels = news['labels']
test = news['testdata'].toarray()
testlabels = news['testlabels']

# Calculate pi and group cuts
group_cuts = []
test_cuts = []
group_cuts.append(0)
test_cuts.append(0)
for x in range(1, 21):
	count = np.count_nonzero(labels==x)
	group_cuts.append(group_cuts[x-1] + count)
	count = np.count_nonzero(testlabels==x)
	test_cuts.append(test_cuts[x-1] + count)

# Set up negative and positive groups.
negative = [1, 16, 20]
positive = [17, 18, 19]
train_negative_count = 0
test_negative_count = 0
train_positive_count = 0
test_positive_count = 0

# Modify the class labels and count the numbers
for x in negative:
	labels[group_cuts[x-1]:group_cuts[x]] = [0]
	train_negative_count += group_cuts[x] - group_cuts[x-1]
	testlabels[test_cuts[x-1]:test_cuts[x]] = [0]
	test_negative_count += test_cuts[x] - test_cuts[x-1]

for x in positive:
	labels[group_cuts[x-1]:group_cuts[x]] = [1]
	train_positive_count += group_cuts[x] - group_cuts[x-1]
	testlabels[test_cuts[x-1]:test_cuts[x]] = [1]
	test_positive_count += test_cuts[x] - test_cuts[x-1]

# Sort labels and choose labels with only 0 and 1.
train_index = np.squeeze(np.argsort(labels, axis = 0))[:train_positive_count+train_negative_count]
test_index = np.squeeze(np.argsort(testlabels, axis = 0))[:test_positive_count+test_negative_count]

pi = []
pi.append(train_negative_count/float(train_negative_count+train_positive_count))
pi.append(train_positive_count/float(train_negative_count+train_positive_count))

train_cuts = []
train_cuts.append(0)
train_cuts.append(train_negative_count)
train_cuts.append(train_negative_count+train_positive_count)

# Calculate mius according to defined estimator using Laplace smoothing
def cal_laplace_miu(data, cuts):
	mius = []
	for x in range(1, 3):
		miu = (1 + np.sum(data[cuts[x-1]: cuts[x]], axis = 0)) / (2 + cuts[x] - cuts[x-1])
		mius.append(miu)
	return np.array(mius).reshape(2, len(data[0]))

# Calculate the class label for pass in data set.
def classify(data, mius, pi):
	log_pi = np.log(pi)
	subtracted_data = 1 - data
	subtracted_mius = 1 - mius
	log_mius = np.log(mius)
	log_subtracted_mius = np.log(subtracted_mius)

	xlogmiu = data.dot(log_mius.T)
	subtract_part = subtracted_data.dot(log_subtracted_mius.T)
	log_pi_tiled = np.tile(log_pi.T, (len(data), 1))

	sums = xlogmiu + subtract_part + log_pi_tiled

	# Labels starts from 0.
	return np.argmax(sums, axis = 1)

def cal_error_rate(pred, actual):
	return np.count_nonzero(pred-actual)/float(len(actual))

mius = cal_laplace_miu(train[train_index], train_cuts)

#print log_mius.shape

print "Begin to calculate train error rate"
pred = classify(train[train_index], mius, pi)
print "error rate for train is: %f" % cal_error_rate(pred, labels[train_index].flatten())

print "Begin to calculate test data error rate"
pred = classify(test[test_index], mius, pi)
print "error rate for test is: %f" % cal_error_rate(pred, testlabels[test_index].flatten())


# Codes used to produce word list. 
# No need to submit so comment out.

#subtracted_mius = 1 - mius
#print mius
#log_mius = np.log(mius)
#log_subtracted_mius = np.log(subtracted_mius)

#res = log_mius - log_subtracted_mius
#alefa = res[1] - res[0]

#index = np.argsort(alefa, axis = 0)
#small_ten = index[:20]
#big_ten = index[len(index)-20:]

#lines = [line.rstrip('\n') for line in open('news.vocab')]
#for x in big_ten:
#	print lines[x]
#print "small:"
#for x in small_ten:
#	print lines[x]




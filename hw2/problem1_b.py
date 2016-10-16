from scipy.io import loadmat
import numpy as np 

# Load data and labels
news = loadmat('news.mat')
train = news['data'].toarray()
labels = news['labels']
test = news['testdata'].toarray()
testlabels = news['testlabels']

# Calculate pi and group cuts
group_cuts = []
group_cuts.append(0)
pi = []
for x in range(1, 21):
	count = np.count_nonzero(labels==x)
	pi.append(count / float(len(labels)))
	group_cuts.append(group_cuts[x-1] + count)

# Calculate mius according to defined estimator using Laplace smoothing
def cal_laplace_miu(data, cuts):
	mius = []
	for x in range(1, 21):
		miu = (1 + np.sum(data[cuts[x-1]: cuts[x]], axis = 0)) / (2 + cuts[x] - cuts[x-1])
		mius.append(miu)
	return np.array(mius).reshape(20, len(data[0]))

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

	# Labels starts from 1.
	return np.argmax(sums, axis = 1)+1 

def cal_error_rate(pred, actual):
	return np.count_nonzero(pred-actual)/float(len(actual))


# Calculate train error_rate:
mius = cal_laplace_miu(train, group_cuts)
print "Begin to calculate train error rate"
pred = classify(train, mius, pi)
print "error rate for train is: %f" % cal_error_rate(pred, labels.flatten())

print "Begin to calculate test data error rate"
pred = classify(test, mius, pi)
print "error rate for test is: %f" % cal_error_rate(pred, testlabels.flatten())








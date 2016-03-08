from matplotlib import pyplot

'''data_train = ([], [])
data_test = ([], [])
filename = "result1"
with open(filename+".txt", "r") as f:
	epoch_num = 1
	for line in f:
		line_parts = line.strip().split("\t")
		if line_parts[0].startswith('Epoch'):
			x, y = data_train
			x.append(epoch_num)
			accuracy = line_parts[2].split(" ")
			y.append(accuracy[1])
			epoch_num = epoch_num + 1 
		elif line_parts[0].startswith('Test'):
			x, y = data_test
			x.append(epoch_num)
			accuracy = line_parts[1].split(" ")
			y.append(accuracy[1])

figure = pyplot.figure()

trainx = data_train[0]
trainy = data_train[1]
testx = data_test[0]
testy = data_test[1]

fig_axs = figure.add_axes((.1, .4, .8, .5))

train_trace, = fig_axs.plot(trainx, trainy, label='Training')
test_trace, = fig_axs.plot(testx, testy, label='Test')

fig_axs.grid(b=True, which='both')

pyplot.xlim(min(min(trainx), min(testx)), max(max(trainx), max(testx)))
pyplot.ylim(0.0, 1.0)

pyplot.legend(handles=[train_trace, test_trace], labels=['Training', 'Test'])

axes = pyplot.gca()
axes.set_xlabel('Epochs')
axes.set_ylabel('Accuracy')
txt = "empty text"
#axes.bar(.2, .5)
figure.text(.1, .1, txt)


figure.savefig(filename+".jpg")

pyplot.show()
'''


def plot(trainx, testx, trainy, testy, fname, xlabel, ylabel, ylim, txt):
	figure = pyplot.figure(fname)
	fig_axs = figure.add_axes((.1, .4, .8, .5))

	train_trace, = fig_axs.plot(trainx, trainy, label='Training')
	test_trace, = fig_axs.plot(testx, testy, label='Test')

	fig_axs.grid(b=True, which='both')

	pyplot.xlim(min(min(trainx), min(testx)), max(max(trainx), max(testx)))
	pyplot.ylim(0.0, ylim)

	pyplot.legend(handles=[train_trace, test_trace], labels=['Training', 'Test'])

	axes = pyplot.gca()
	axes.set_xlabel(xlabel)
	axes.set_ylabel(ylabel)

	figure.text(.1, .1, txt)
	figure.savefig(fname)

import collections
import datetime
import numpy
import os
import random
import sys
import theano
import time

from gensim.models import Word2Vec

from myplot import plot

floatX=theano.config.floatX
plot_caption = ""

def update_plot_caption(name, value):
    global plot_caption
    plot_caption = plot_caption + name + ": " + str(value) + "\n"

class RnnClassifier(object):
    def __init__(self, n_words, n_classes, word2id):
        # network parameters
        random_seed = 42
        word_embedding_size = 300
        recurrent_size = 100
        l2_regularisation = 0.0001

        update_plot_caption("random_seed", random_seed)
        update_plot_caption("word_embedding_size", word_embedding_size)
        update_plot_caption("recurrent_size", recurrent_size)
        update_plot_caption("l2_regularisation", l2_regularisation)

        # random number generator
        self.rng = numpy.random.RandomState(random_seed)

        # this is where we keep shared weights that are optimised during training
        self.params = collections.OrderedDict()

        # setting up variables for the network
        input_indices = theano.tensor.ivector('input_indices')
        target_class = theano.tensor.iscalar('target_class')
        learningrate = theano.tensor.fscalar('learningrate')

        # creating the matrix of word embeddings
        
        word_embeddings = self.create_word_vectors('word_embeddings', (n_words, word_embedding_size), word2id)
        theano.pp(word_embeddings)

        # extract the relevant word embeddings, given the input word indices
        input_vectors = word_embeddings[input_indices]

        # gated recurrent unit
        # from: Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation (Cho et al, 2014)
        def gru_step(x, h_prev, W_xm, W_hm, W_xh, W_hh):
            m = theano.tensor.nnet.sigmoid(theano.tensor.dot(x, W_xm) + theano.tensor.dot(h_prev, W_hm))
            r = _slice(m, 0, 2)
            z = _slice(m, 1, 2)
            _h = theano.tensor.tanh(theano.tensor.dot(x, W_xh) + theano.tensor.dot(r * h_prev, W_hh))
            h = z * h_prev + (1.0 - z) * _h
            return h

        W_xm = self.create_parameter_matrix('W_xm', (word_embedding_size, recurrent_size*2))
        W_hm = self.create_parameter_matrix('W_hm', (recurrent_size, recurrent_size*2))
        W_xh = self.create_parameter_matrix('W_xh', (word_embedding_size, recurrent_size))
        W_hh = self.create_parameter_matrix('W_hh', (recurrent_size, recurrent_size))
        initial_hidden_vector = theano.tensor.alloc(numpy.array(0, dtype=floatX), recurrent_size)

        hidden_vector, _ = theano.scan(
            gru_step,
            sequences = input_vectors,
            outputs_info = initial_hidden_vector,
            non_sequences = [W_xm, W_hm, W_xh, W_hh]
        )
        hidden_vector = hidden_vector[-1]

        # hidden->output weights
        W_output = self.create_parameter_matrix('W_output', (n_classes,recurrent_size))
        output = theano.tensor.nnet.softmax([theano.tensor.dot(W_output, hidden_vector)])[0]
        predicted_class = theano.tensor.argmax(output)

        # calculating the cost function
        cost = -1.0 * theano.tensor.log(output[target_class])
        for m in self.params.values():
            cost += l2_regularisation * (theano.tensor.sqr(m).sum())

        # calculating gradient descent updates based on the cost function
        gradients = theano.tensor.grad(cost, self.params.values())
        updates = [(p, p - (learningrate * g)) for p, g in zip(self.params.values(), gradients)]

        # defining Theano functions for training and testing the network
        self.train = theano.function([input_indices, target_class, learningrate], [cost, predicted_class], updates=updates, allow_input_downcast = True)
        self.test = theano.function([input_indices, target_class], [cost, predicted_class], allow_input_downcast = True)

    def create_parameter_matrix(self, name, size):
        """Create a shared variable tensor and save it to self.params"""
        vals = numpy.asarray(self.rng.normal(loc=0.0, scale=0.1, size=size), dtype=floatX)
        self.params[name] = theano.shared(vals, name)
        return self.params[name]
    def create_word_vectors(self, name, size, word2id):
        model = Word2Vec.load_word2vec_format("/Users/shushan/word2vec/word2vec-trimmed.bin", binary=True)
        vals = numpy.asarray(self.rng.normal(loc=0.0, scale=0.1, size=size), dtype=floatX)
        i = 0
        for word in word2id:
            try:
                #vals[i] = model[word]
                pass
            except Exception, e:
                pass 
            i = i + 1 
        self.params[name] = theano.shared(vals, name)
        return self.params[name]

def _slice(M, slice_num, total_slices):
    """ Helper function for extracting a slice from a tensor"""
    if M.ndim == 3:
        l = M.shape[2] / total_slices
        return M[:, :, slice_num*l:(slice_num+1)*l]
    elif M.ndim == 2:
        l = M.shape[1] / total_slices
        return M[:, slice_num*l:(slice_num+1)*l]
    elif M.ndim == 1:
        l = M.shape[0] / total_slices
        return M[slice_num*l:(slice_num+1)*l]

def get_level_from_string(string):
    if string=="A1":
        return 0
    if string=="A2":
        return 1
    if string=="B1":
        return 2
    if string=="B2":
        return 3
    if string=="C1":
        return 4
    if string=="C2":
        return 5
    raise Exception("Unexpected level value: " + string)   

def read_dataset(path):
    """Read a dataset, where the first column contains a real-valued score,
    followed by a tab and a string of words.
    """
    #print "printing line_parts"
    dataset = []
    with open(path, "r") as f:
        for line in f:
            line_parts = line.strip().split(",")
            #print str(line_parts)
            dataset.append((line_parts[0], line_parts[1]:[-1], get_level_from_string(line_parts[2])))
    #print str(dataset)
    return dataset

def score_to_class_index(score, n_classes):
    """Maps a real-valued score between [0.0, 1.0] to a class id, given n_classes."""
    for i in xrange(n_classes):
        if score <= (i + 1.0) * (1.0 / float(n_classes)):
            return i

def create_dictionary(sentences, min_freq):
    """Creates a dictionary that maps words to ids.
    If min_freq is positive, removes all words that have a smaller frequency.
    """
    counter = collections.Counter()
    for sentence in sentences:
        for word in sentence:
            counter.update([word])

    word2id = collections.OrderedDict()
    word2id["<unk>"] = 0
    word2id["<s>"] = 1
    word2id["</s>"] = 2

    word_count_list = counter.most_common()
    for (word, count) in word_count_list:
        if min_freq < 0 or count >= min_freq:
            word2id[word] = len(word2id)

    return word2id

def sentence2ids(words, word2id):
    """Takes a list of words and converts them to ids using the word2id dictionary."""
    ids = [word2id["<s>"],]
    for word in words:
        if word in word2id:
            ids.append(word2id[word])
        else:
            ids.append(word2id["<unk>"])
    ids.append(word2id["</s>"])
    return ids

if __name__ == "__main__":
    path_train = sys.argv[1]
    path_test = sys.argv[2]

    # training parameters
    min_freq = -1
    epochs = 25
    learningrate = 0.01
    n_classes = 6

    # params for plot caption
    update_plot_caption("min_freq", min_freq)
    update_plot_caption("epochs", epochs)
    update_plot_caption("learningrate", learningrate)
    update_plot_caption("n_classes", n_classes)

    # reading the datasets
    sentences_train = read_dataset(path_train)
    sentences_test = read_dataset(path_test)

    # creating the dictionary from the training data
    word2id = create_dictionary([(word + " " + definition).split() for word, definition, level in sentences_train], min_freq)

    # mapping training and test data to the dictionary indices
    data_train = [(level, sentence2ids((word + " " + definition).split(), word2id)) for word, definition, level in sentences_train]
    data_test = [(level, sentence2ids((word + " " + definition).split(), word2id)) for word, definition, level in sentences_test]

    # shuffling the training data
    random.seed(1)
    random.shuffle(data_train)

    # creating the classifier
    rnn_classifier = RnnClassifier(len(word2id), n_classes, word2id)

    # open file to write ouput
    ts = time.time()
    st = datetime.datetime.fromtimestamp(ts).strftime('%Y%m%d-%H%M%S')
    pdir = "results/" + st

    os.makedirs(pdir)
    pdir += "/"
    output_filename = pdir + st + ".txt"

    with open(output_filename, "w") as f:
        # initializing lists for plotting
        x = range(1, epochs+1)

        acc_train = []
        acc_test = []
        
        cost_train = []
        cost_test = []
        
        mse_train = []
        mse_test = []

        for epoch in xrange(epochs):
            # training
            cost_sum = 0.0
            correct = 0
            mse = 0
            for target_class, sentence in data_train:
                cost, predicted_class = rnn_classifier.train(sentence, target_class, learningrate)
                cost_sum += cost
                mse += (predicted_class - target_class)*(predicted_class - target_class)
                if predicted_class == target_class:
                    correct += 1

            f.write("Epoch: " + str(epoch) + "\tCost: " + str(cost_sum) + "\tAccuracy: " + str(float(correct)/len(data_train)) + "\tMSE: " + str(float(mse)/len(data_train)) + "\n")
            f.flush()
            # saving train data for plotting
            acc_train.append(float(correct)/len(data_train))
            cost_train.append(cost_sum)
            mse_train.append(float(mse)/len(data_train))

            # testing
            cost_sum = 0.0
            correct = 0
            mse = 0
            
            for target_class, sentence in data_test:
                cost, predicted_class = rnn_classifier.test(sentence, target_class)
                cost_sum += cost
                mse += (predicted_class - target_class)*(predicted_class - target_class)
                if predicted_class == target_class:
                    correct += 1
            
            f.write("Test_cost: " + str(cost_sum) + "\tTest_accuracy: " + str(float(correct)/len(data_test)) + "\tMSE: " + str(float(mse)/len(data_test)) + "\n")
            f.flush()
            # saving test data for plotting
            acc_test.append(float(correct)/len(data_test))
            cost_test.append(cost_sum)
            mse_test.append(float(mse)/len(data_test))    
        f.close()
        
        #plotting results
        xlabel = "Epochs"
        #plot accuracy
        acc_ylabel = "Accuracy"
        acc_fname = pdir+st+"-accuracy.jpg"
        acc_ylim = 1.0
        plot(x, x, acc_train, acc_test, acc_fname, xlabel, acc_ylabel, acc_ylim, plot_caption)
        
        #plot costs
        cost_ylabel = "Cost"
        cost_fname = pdir+st+"-cost.jpg"
        cost_ylim = max(max(cost_train), max(cost_test))
        plot(x, x, cost_train, cost_test, cost_fname, xlabel, cost_ylabel, cost_ylim, plot_caption)
        
        #plot mean squared errors
        mse_ylabel = "Mean Squared Error"
        mse_fname = pdir+st+"-mse.jpg"
        mse_ylim = 25.0
        plot(x, x, mse_train, mse_test, mse_fname, xlabel, mse_ylabel, mse_ylim, plot_caption)
        



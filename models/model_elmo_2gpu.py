import tensorflow as tf
import tensorflow_hub as hub

class Model_elmo:
    def __init__(self, args, textData):
        print('Creating Bilstm Model')

        self.args = args
        self.textData = textData

        self.dropOutRate = None
        self.initial_state = None
        self.learning_rate = None
        self.loss = None
        self.optOp = None
        self.labels = None
        self.input = None
        self.target = None
        self.length = None
        self.embedded = None
        self.predictions = None
        self.batch_size = None
        self.corrects = None


        self.v0 = None
        self.v1 = None
        self.v2 = None
        self.v3 = None
        self.v4 = None
        self.v5 = None
        self.v6 = None
        self.v7 = None

        self.buildNetwork()

    def buildNetwork(self):
        with tf.name_scope('rnn'):
            # [batchSize, hiddenSize*2]
            outputs = self.buildRNN()

        with tf.name_scope('output'):
            weights = tf.get_variable(name='weights', shape=[self.args.hiddenSize*4, self.args.numClasses],
                                      initializer=tf.contrib.layers.xavier_initializer())

            biases = tf.get_variable(name='biases', shape=[self.args.numClasses],
                                      initializer=tf.contrib.layers.xavier_initializer())
            # [batchSize, numClasses]
            logits = tf.nn.xw_plus_b(x=outputs, weights=weights, biases=biases)
        with tf.name_scope('predictions'):
            # [batchSize]
            self.predictions = tf.argmax(logits, axis=-1, name='predictions', output_type=tf.int32)
            # single number
            self.corrects = tf.reduce_sum(tf.cast(tf.equal(self.predictions, self.labels), tf.int32), name='corrects')

        with tf.name_scope('loss'):
            loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=self.labels, name='loss')

            self.loss = tf.reduce_sum(loss)

        with tf.name_scope('backpropagation'):
            opt = tf.train.AdamOptimizer(learning_rate=self.args.learningRate, beta1=0.9, beta2=0.999,
                                               epsilon=1e-08)
            self.optOp = opt.minimize(self.loss)

    def buildRNN(self):
        with tf.name_scope('placeholders'):
            # [batchSize, maxSteps]
            input_shape = [None, self.args.maxSteps]
            self.input = tf.placeholder(tf.string, shape=input_shape, name='input')
            self.labels = tf.placeholder(tf.int32, shape=[None,], name='labels')
            self.length = tf.placeholder(tf.int32, shape=[None,], name='length')
            self.batch_size = tf.placeholder(tf.int32, shape=(), name='batch_size')

            self.dropOutRate = tf.placeholder(tf.float32, (), name='dropOut')

        with tf.name_scope('elmo_embedding_layer'):
            print('Using elmo word embeddings!')
            elmo = hub.Module("https://tfhub.dev/google/elmo/2", trainable=True)
            with tf.device('/gpu:1'):
                self.embedded = elmo(
                    inputs={
                        "tokens": self.input,
                        "sequence_len": self.length
                    },
                    signature="tokens",
                    as_dict=True)['elmo']

                # [batchSize, maxSteps, embeddingSize]
                self.embedded = tf.nn.dropout(self.embedded, self.dropOutRate, name='embedding_dropout')

        with tf.name_scope('lstm'):
            with tf.variable_scope('cell', reuse=False):

                def get_cell(hiddenSize, dropOutRate):
                    cell = tf.contrib.rnn.LSTMCell(num_units=hiddenSize, state_is_tuple=True,
                                                   initializer=tf.contrib.layers.xavier_initializer())
                    cell = tf.contrib.rnn.DropoutWrapper(cell, input_keep_prob=dropOutRate,
                                                             output_keep_prob=dropOutRate)
                    return cell

                # https://stackoverflow.com/questions/47371608/cannot-stack-lstm-with-multirnncell-and-dynamic-rnn
                multiCell_fw = []
                for i in range(self.args.rnnLayers):
                    multiCell_fw.append(get_cell(self.args.hiddenSize, self.dropOutRate))
                multiCell_fw = tf.contrib.rnn.MultiRNNCell(multiCell_fw, state_is_tuple=True)

                multiCell_bw = []
                for i in range(self.args.rnnLayers):
                    multiCell_bw.append(get_cell(self.args.hiddenSize, self.dropOutRate))
                multiCell_bw = tf.contrib.rnn.MultiRNNCell(multiCell_bw, state_is_tuple=True)

            # [batchSize, maxSteps, hiddenSize]
            outputs, state = tf.nn.bidirectional_dynamic_rnn(cell_fw=multiCell_fw, cell_bw=multiCell_bw,
                                inputs=self.embedded, sequence_length=self.length, dtype=tf.float32)

            outputs_fw = outputs[0]
            outputs_bw = outputs[1]


            # [batchSize, maxSteps]
            last_relevant_mask = tf.one_hot(indices=self.length-1, depth=self.args.maxSteps, name='last_relevant',
                                            dtype=tf.int32)
            # [batchSize, hiddenSize]
            last_relevant_outputs_fw = tf.boolean_mask(outputs_fw, last_relevant_mask, name='last_relevant_outputs_fw')
            last_relevant_outputs_bw = tf.boolean_mask(outputs_bw, last_relevant_mask, name='last_relevant_outputs_bw')

        with tf.variable_scope('attention', reuse=tf.AUTO_REUSE):
            def build_attention(from_vector, to_vector, to_vector_length):
                '''
                Similar attention mechanism to https://arxiv.org/pdf/1409.0473.pdf

                :param from_vector: [batchSize, hiddenSize]
                :param to_vector: [batchSize, maxSteps, hiddenSize]
                :param to_vector_length: [batchSize]
                :return: att_vec: [batchSize, hiddenSize]
                '''

                # [batchSize, 1, hiddenSize]
                from_vector = tf.expand_dims(from_vector, axis=1)

                # [batchSize, maxSteps, hiddenSize]
                from_vector = tf.tile(from_vector, multiples=[1, self.args.maxSteps, 1])

                # [hiddenSize, attSize]
                weights_from = tf.get_variable(shape=[self.args.hiddenSize, self.args.attSize], name='weights_from')
                weights_to = tf.get_variable(shape=[self.args.hiddenSize, self.args.attSize], name='weights_to')

                # [batchSize*maxSteps, attSize]
                from_vector_ = tf.matmul(tf.reshape(from_vector, shape=[-1, self.args.hiddenSize]),
                                        weights_from, name='from_vector_')
                to_vector_ = tf.matmul(tf.reshape(to_vector, shape=[-1, self.args.hiddenSize]),
                                      weights_to, name='to_vector_')

                # [batchSize*maxSteps, attSize]
                output_vec = tf.tanh(from_vector_+to_vector_, name='output_vec')

                weights_project = tf.get_variable(shape=[self.args.attSize, 1], name='weights_project')

                # [batchSize*maxSteps, 1]
                logits = tf.matmul(output_vec, weights_project)

                # [batchSize*maxSteps]
                logits = tf.squeeze(logits, axis=-1)

                # [batchSize, maxSteps]
                logits = tf.reshape(logits, shape=[-1, self.args.maxSteps], name='logits')
                # [batchSize, maxSteps]
                mask = tf.sequence_mask(lengths=to_vector_length, maxlen=self.args.maxSteps, dtype=tf.float32)
                mask = tf.log(mask, name='mask')

                # [batchSize, maxSteps]
                logits_masked = tf.add(logits, mask, name='logits_masked')

                # [batchSize, maxSteps]
                # alpha is 0 for invalid time steps
                alpha = tf.nn.softmax(logits_masked, axis=-1, name='alpha')
                # [batchSize, maxSteps, 1]
                alpha = tf.expand_dims(alpha, axis=-1)

                # [batchSize, maxSteps, hiddenSize]
                alpha = tf.tile(alpha, multiples=[1, 1, self.args.hiddenSize])
                # multiply each time step with its corresponding weights
                # [batchSize, maxSteps, hiddenSize]
                att_vec = tf.multiply(to_vector, alpha)

                # [batchSize, hiddenSize]
                att_vec = tf.reduce_sum(att_vec, axis=1, name='att_vec')

                return att_vec

            # the last relevant output attend to its previous outputs (do not include the last relevant output itself!)
            # [batchSize, hiddenSize]

            attention_vec_fw = build_attention(from_vector=last_relevant_outputs_fw,
                                               to_vector=outputs_fw, to_vector_length=self.length-1)

            attention_vec_bw = build_attention(from_vector=last_relevant_outputs_bw,
                                               to_vector=outputs_bw, to_vector_length=self.length-1)


            # [batchSize, hiddenSize*2]
            outputs_vec_fw = tf.concat(values=[attention_vec_fw, last_relevant_outputs_fw], axis=-1, name='outputs_vec_fw')
            outputs_vec_bw = tf.concat(values=[attention_vec_bw, last_relevant_outputs_bw], axis=-1, name='outputs_vec_bw')

            # [batchSize, hiddenSize*4]
            outputs_vec = tf.concat(values=[outputs_vec_fw, outputs_vec_bw], axis=-1, name='outputs_vec')

        return outputs_vec

    def step(self, batch, test=False):
        feed_dict = {}

        # [batchSize, maxSteps]
        input_ = []
        length = []
        labels = []

        for sample in batch.samples:
            input_.append(sample.sentence)
            labels.append(sample.label)
            length.append(sample.length)

        feed_dict[self.labels] = labels
        feed_dict[self.input] = input_
        feed_dict[self.length] = length
        feed_dict[self.batch_size] = len(length)

        if not test:
            feed_dict[self.dropOutRate] = self.args.dropOut
            ops = (self.optOp, self.loss, self.predictions, self.corrects)
        else:
            # during test, do not use drop out!!!!
            feed_dict[self.dropOutRate] = 1.0
            ops = (self.loss, self.predictions, self.corrects)

        return ops, feed_dict


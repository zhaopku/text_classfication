class Sample:
    def __init__(self, data, words, steps, label, flag_word):
        self.input_ = data[0:steps]
        self.sentence = words[0:steps]
        self.length = 0
        self.label = label

        for word in self.input_:
            if word == flag_word:
                break
            self.length += 1

class Batch:
    def __init__(self, samples):
        self.samples = samples
        self.batch_size = len(samples)

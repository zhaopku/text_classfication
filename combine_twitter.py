import os

def combine(tag='train'):
    neg_file = open(os.path.join('data', tag+'_neg_splited.txt'), 'r')
    pos_file = open(os.path.join('data', tag+'_pos_splited.txt'), 'r')

    neg_samples = neg_file.readlines()
    pos_samples = pos_file.readlines()

    assert len(pos_samples) == len(neg_samples)

    samples = []
    for idx, line in enumerate(pos_samples):
        pos = line.strip()
        neg = neg_samples[idx].strip()
        samples.append(pos+'\t'+'1')
        samples.append(neg+'\t'+'0')

    print('Number of {} samples'.format(tag))

    with open(os.path.join('data', tag+'_splited.txt'), 'w') as out:
        for sample in samples:
            out.write(sample+'\n')

if __name__ == '__main__':
    combine('train')
    combine('val')

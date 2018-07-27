import os
import random
src_data_dir = 'twitter-datasets'
target_data_dir = 'data'

file_neg = 'train_neg_full.txt'
file_pos = 'train_pos_full.txt'

# note: use 10% of the data as validation set, the remaining as training
val_ratio = 0.1

def split_data(file_name, out_file_name):
    file_name = os.path.join(src_data_dir, file_name)
    with open(file_name, 'r') as file:
        lines = file.readlines()
        random.shuffle(lines)

        num_train = int(len(lines) * (1-val_ratio))

        train_lines = lines[:num_train]

        val_lines = lines[num_train:]

    with open(os.path.join(target_data_dir, 'train_' + out_file_name), 'w') as file:
        for line in train_lines:
            file.write(line)

    with open(os.path.join(target_data_dir, 'val_' + out_file_name), 'w') as file:
        for line in val_lines:
            file.write(line)

if __name__ == '__main__':
    split_data(file_neg, 'neg_splited.txt')
    split_data(file_pos, 'pos_splited.txt')
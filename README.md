# ETH, Computational Intelligence Lab. Text Classification Task, 2018

## Requirements 
    1. python3
    2. TensorFlow 1.8
    3. tensorflow-hub (pip install tensorflow-hub)
    4. nltk (for word tokenization)
    5. tqdm (for progress bar)

## Usage
	python3 main.py [commandline options]
       (see details of the commandline options at models/predict.py)

## Train
    1. python3 main.py --hiddenSize 512 --rnnLayers 2 --maxSteps 50 --attSize 512 --dropOut 0.8 --batchSize 200 --vocabSize -1 --elmo

## Reproduce best result
    1. Download the trained model and the pkl file at https://polybox.ethz.ch/index.php/s/AyZUvlTwotdd1Jr
    2. unzip at the root directory of the project 
       (please use the 'unzip' command in the terminal, as directly unzipping in GUI might create an additional parent folder in macOS)
    3. python3 main.py --hiddenSize 512 --rnnLayers 2 --maxSteps 50 --attSize 512 --dropOut 0.8 --batchSize 200 --vocabSize -1 --elmo --testModel --loadModel
    4. The final output prediction is in the dir ./test_result

## Attention
    1. Please use the EXACT pkl file in the zip file downloaded.
       The pkl file is created to avoid processing the original dataset file each time before training. Since there are
       some randomness when creating the pkl file, which would affect word index, hence please use EXACTLY the pkl file downloaded
       to reproduce the result.
    2. create_data.py and combine_twitter.py are used to process the dataset for easier processing.
       Unzip the dataset at the root directory, execute create_data.py and then combine_twitter.py to create your own dataset.
       DO NOT USE THE NEWLY CREATED DATASET FOR REPPRODUCING THE RESULT.
    3. Please be patient while reproduing the result. It takes about 20 minutes on a 2.6Ghz i7 with 16GB RAM machine.

## Contact
    Zhao Meng, zhmeng@student.ethz.ch
    Siwei Zhang, szhang@student.ethz.ch
    Kangning Liu, liuka@student.ethz.ch

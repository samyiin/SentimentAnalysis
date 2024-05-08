# Sentiment Analysis
## Description
In this project, I am trying to train a model that can understand sentiment of a sentence. I will try to train on several models, and see who is performing better. 
This is a project that extends from the project I did in Hebrew University course 67658 Natural Language Processing. 
I optimized the design of DataManger, and train the data on more models for learning purposes. 

## Data 
I get my data from the Standardford Sentiment Tree Bank (SST). You can download from this website (Under "Dataset Downloads:"

    https://nlp.stanford.edu/sentiment/

## DataLoader
We have to write our own data loader to load the dataset into tree like structure, and clean up some redundent symbols. 
This is not the DataLoader that feeds data to models to train, this is the dataloader that read raw data from SST and parse it into python objects. 
Then we write a class called Online DataLoader, that inherit from pyTorch DataLoader class, it will embed sentences to certain representation given the embedding function. 

## Embedding
We will mainly use word2Vec as our representation of a word. But it is possible that in the future, we will experiment with OpenAI's embedding services. 
For Word2Vec, we downloaded the original 3-million word embeddings, but in this project, we are going to crop it into dictionary that contains only words in the dataset. 

## Training
### RNN
The first thing we tried, is to train this dataset on an RNN. The task is, given the sentence, we will rpedict the sentiment of the sentence. 
A paper describing RNN can be found under the Papers/ folder. 

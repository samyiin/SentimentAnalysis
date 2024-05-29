# Sentiment Analysis
## Description
In this project, I am trying to train a model that can understand sentiment of a sentence. I 
This is a project that extends from the project I did in Hebrew University course 67658 Natural Language Processing. 
I optimized the design of DataManger, and train the data on more models for learning purposes. 

## Data 
I get my data from the Standardford Sentiment Tree Bank (SST). You can download from this website (Under "Dataset Downloads:"

    https://nlp.stanford.edu/sentiment/

## DataLoader
We have to write our own data loader to load the dataset into tree like structure, and clean up some redundent symbols. 
This is not the DataLoader that feeds data to models to train, this is the dataloader that read raw data from SST and parse it into python objects. 
Then we write a class called Online DataLoader, that inherit from pyTorch DataLoader class, it will embed sentences to certain representation given the embedding function. 

## Representation
We will mainly use word2Vec as our representation of a word. But it is possible that in the future, we will experiment with OpenAI's embedding services. 
For Word2Vec, we downloaded the original 3-million word embeddings, but in this project, we are going to crop it into dictionary that contains only words in the dataset. 
So a sentence is represented by a list of vectors, each vector is the embedding of a word. 

## Training
I trained the corpus on the LSTM model. It's in the BILSTM.ipynb, and there are several things to take heed of, I spent a while trying to find the bug, and I wrote the findings in the notebook. Worth checking it out if you are trying to run a bidirectional LSTM model. 

## Notes
Also, there is a little problem with the gensim package, so my solution is to downgrade to python 3.10 so that I can download a different version of scipy (a denpendency of gensim). It's written in the example.ipynb notebook, also worth checking it out when you are trying to setup the environment. 

## Results
I trained 20 epoch on the data, and we can see that the accuracy of the validation set reach maximum at the 7-8'th epoch. After that, the training accuray of the model keeps increasing, but the generalization accuracy (accuracy of validation) is decreasing, that means we have an overfit. 
Eventually, the model have around 81% accuracy in predicting the sentiment of a sentence. 

## Future Direction
I can try different representation of the words, such as the openAI embedding. And also, I can explore other nn structures. And I might be able to train on the sentiment score (float) instead of the binary sentiment value. However, as the current technology shifted towards large language models, and LLM performs much better than these toy examples, I guess in a short while there will be no updates on this projects. 

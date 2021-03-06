# stackoverflow.com-answer-predictor
To predict expected answer for a stackoverflow.com question.

***Introduction-***
Question and Answer (Q&A) sites help developers dealing with the 
increasing complexity of software systems and third-party components by 
providing a platform for exchanging knowledge about programming topics. A 
shortcoming of Q&A sites is that they provide no indication on what could be 
potential answer. Such an indication would help, for example, the developers 
who posed the questions in managing their time. It also could be very 
discouraging to young developers if they don’t have their doubts cleared. We 
will try to fill this gap by investigating whether and how answer for a 
question posted on StackOverflow, a prominent example of Q&A website, can 
be predicted. To fulfill this aim, we will first determine the types of answers 
to be considered valid answers to the question, after which the best answer 
answer is predicted.

In this project work, we are trying to predict the answer of a prospective 
question entered by a user on StackOverflow. We are going to use different 
data mining algorithms and will analyze by comparing their results. The main 
tasks to perform for this project are parsing of the questions tags, train the 
system based on tags and body and test the system with remaining data. We 
will predict accuracy of an algorithm by comparing the predicted answer to 
get a response against the existing answer from test data. This way we can 
decide the accuracy, precision, recall and the other metrics for an algorithm. 
The overall aim of a project is to have an idea of what could be possible 
answer on ​ stackoverflow.com​ . 

The features of a good answer considered by us are:-
● ***Non-Stopwords*** 
● ***No. of occurences of a word in given text*** 
● ***Relevance between the answerBody, questionBody and questionTags*** 
● ***Information provided by the data*** 
● ***Unique words in the text*** 
● ***Answer Subjectivity*** 
● ***Answer Score*** 
● ***Answer Upvotes*** 
● ***Answer Downvotes*** 
● ***Answerer Reputation*** 
● ***Answer Comment Count*** 
● ***Readability consensus of answer***

To reproduce the work run these Python scripts in order-<br/>
```
 ~$ python3 csvDatasetPreprocessing.py #Preprocess CSV Dataset
 
 ~$ python3 FeatureVectorsGeneration.py #generate the feature vectors
 
 ~$ python3 Train_2.py #To get the model trained and know mean accuracy score on training set
 
 ~$ python3 Test_2.py #To get the model tested and know mean accuracy score on testing set
 
 ~$ TFIDF_util.py #To create tf-idf vector space of questions title and body for similarity matching
 
 ~$ Predictor.py #To finally input a prospective Stackoverflow.com question and get apt answers
```

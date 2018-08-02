# stackoverflow.com-answer-predictor
To predict expected answer for a stackoverflow.com question.

Introduction-
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



To reproduce the work run these Python scripts in order-

 $ python3 csvDatasetPreprocessing.py #Preprocess CSV Dataset
 
 $ python3 FeatureVectorsGeneration.py #generate the feature vectors
 
 $ python3 Train_2.py #To get the model trained and know mean accuracy score on training set
 
 $ python3 Test_2.py #To get the model tested and know mean accuracy score on testing set
 
 $ TFIDF_util.py #To create tf-idf vector space of questions title and body for similarity matching
 
 $ Predictor.py #To finally input a prospective Stackoverflow.com question and get apt answers

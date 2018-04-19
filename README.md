# stackoverflow.com-answer-predictor
To predict expected answer for a stackoverflow.com question.

To reproduce the work run these Python scripts in order-

 $ python3 csvDatasetPreprocessing.py #Preprocess CSV Dataset
 
 $ python3 FeatureVectorsGeneration.py #Its generate the feature vectors
 
 $ python3 Train_2.py #To get the model trained and know mean accuracy score on training set
 
 $ python3 Test_2.py #To get the model tested and know mean accuracy score on testing set
 
 $ TFIDF_util.py #To create tf-idf vector space of questions title and body for similarity matching
 
 $ Predictor.py #To finally input a prospective Stackoverflow.com question and get apt answers

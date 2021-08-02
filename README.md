<h1 align="center">
         US Census Data
</h1>

<h2 align="center">
         Predicting whether the individual earns over $50k per year from the US Census dataset.
</h2>

<br />

<h2>
          Problem Description:
</h2>
<p align="justify">
The objective of this question is to use the ensemble learning functionality to identify
the extent to which classification performance can be improved through the
combination of multiple models. Experiments will be run on a dataset extracted from
US Census data. The data contains 14 attributes including age, race, sex, marital
status etc, and the goal is to predict whether the individual earns over $50k per year.
</p>

<h2>
         Data Extraction, Cleaning, Wrangling, Standardization: 
</h2>
<p align="justify">
  I did a comprehensive check on the dataset before moving forward with the data cleaning process. Firstly, I imported the dataset in python, and checked for null values in all possible ways and found out that there are 178 missing values in the “workclass” & “occupation” features, and 46 missing values in the “native-country” feature. All of these 3 missing features are categorical. The missing values were not NaN, and were marked as “ ?”, which I specified in the “na_values” parameter of importing the csv file. Furthermore, I noticed that all the categorical feature values and all the column header names had trailing whitespaces throughout the dataset. So, I used the trim() function in python for all the categorical feature values and also the column header names for the whole dataset to make it accessible and well presentable. Now, to handle the missing values I used the “TransformerMixin” function to impute the missing values wherein the “object” dtype are imputed with the mode (most frequent values). I once again used a heatmap to check whether there are any null/missing values, and all the values were handled properly. Now, I further used OrdinalEncoder() on the categorical variables to encode them, since some model classifiers do not take categories(nominal) into consideration. I did not used pandas.getdummies() or OneHotEncoder(), since they tend to bring in more columns and most probably we land into the curse of dimensionality situation, wherein we tend to use PCA for dimensionality reduction, which is was not feasible for my dataset, since there were only 15 columns, I thought about going ahead with OrdinalEncoder(). After tidying up of the fields and categorical encoding, I always make a copy of the dataset, in case I need to use it further again. I also checked for the correlation heatmap, and however I found out that the “fnlwgt” column is not correlated to the class(target) variable, but I thought about keeping it since the dimension of the dataset is already less. I also crosschecked this by applying a quick random forest classifier on it by using the “fnlwgt” and by excluding it from the dataset, and I did not find any significant differences in the evaluation metrics in both the cases, so I decided to keep all the columns and not remove any. I also kept in mind that I have to Standardize the data before applying it on any model, and to also split it into training/testing and further use it for cross validation through a pipelined hyperparameter tuning by using RandomizedSearchCV(). I learnt a lot of new things while performing this extensive task in python, which included, the basics of imputation, label encoding, pipelining, finding the best hyperparameters and many more things which I shall discuss in depth as we go along. In the end, I had a clean and tidy dataset ready to perform various ML algorithms on.
  </p>
  
  <h2>
          Evaluation measure(s):
</h2>
<p align="justify">
  Before evaluating the performance of the given 3 ML Classifiers, I firstly checked the value_counts of the class variable and I found a moderate level of imbalance between both the classes, since it is a binary classification dataset. But the number was not that significant, wherein oversampling or under sampling could’ve been done. Yet, I did implement SMOTE on the training values, but I was not satisfied with the results it was outputting, I will discuss it in the end. Initially, I used train_test_split() on the encoded & tidied up dataset & further standardized it. Although I’m aware that the standardization doesn’t make any difference to Decision tree and Neural Network (MLPClassifier), I did it for the 1-NN Classifier, since it is a distance measured classifier. Further, I used the hyperparameter tuning technique (RandomizedSearchCV) to find the best parameters along with its KFold cross-validation technique. Now, as I’ve kept the dataset originally imbalanced and not used SMOTE or any up/down sampling technique, I used some pretty informative evaluation measures which adhere to the notion of an imbalanced class variable dataset classification. So, after running the Decision Tree, Neural Network, 1-NN classifiers on the dataset, I got the following results, I’ll be discussing about the evaluation measures as we go along! 
  </p>
  
   <p align="center" >
  <img src="https://user-images.githubusercontent.com/47216809/127858772-1f487ee8-b608-4263-90c3-c5dbcd8b68d2.png"  />
  </p>
  
  <p align="justify">
  Now, again, to make it clear, I’ve already encoded, standardized, train_test splitted (test size =0.35) and used the best parameters for these classifiers and have got the above results. As the class variables’ value_counts are imbalanced, I thought about picking the F1 measure along with the Balanced Accuracy and ROC-AUC score. I’ve also included Cohen’s kappa, MCC and Accuracy to just compare the metrics. I chose the ROC-AUC score since I wanted to give equal weight to both the class predictions. And as we want to focus on each of the classes, we look at the precision and recall. And to get a combined effort of both of them, we check the f1-score, which is the harmonic mean of precision and recall. Wherein, for the Neural Networks’ best_params_ I got an f1-score of 0.61, such a low score is expected since the support of one class is greater than another class, but with a balanced accuracy of 0.74, it is the best classifier for the dataset, it also has the highest Cohen Kappa score and MCC score. The balanced accuracy is basically the average of the TP’s and the TN’s of each class individually. Hence it gives us a balanced score taking both the classes in equal consideration. Even though the accuracy of a Neural Network model is the highest from the table above, it is flawed to say that it has outperformed others. The Neural Network classifier is the best here. I bet that the Random Forest will be more effective in terms of the Balanced accuracy. As the accuracy might be skewed across the class distribution, the kappa score can be a best evaluation measure to compute how the classifier performed for all the instances. The Matthews correlation coefficient is also a good evaluator for a skewed binary class distribution, it takes all the 4 confusion matrices elements predictions into consideration. Both the MCC and Cohen’s Kappa score should be high. This is the reason why I chose these evaluation measures for my dataset and predominantly for the imbalanced class distribution along with the model classifiers. I also found out the best value of “k” for k-NN classifier, which was 4 as per the SearchCV with an increased auc-roc score of .8 % from the 1-NN classifier. To wind up, I’ll say that the Neural Network Classifier outperformed the others in this initial stage, when we take the Balanced Accuracy, ROC-AUC, Kappa and MCC score into consideration. We’ll further see how ensembles can improve the model performance of other classifiers.
  </p>
    
   <h2>
          Classification performance: 
</h2>
<p align="justify">
  To start with; Ensemble Learning is an ML technique which is used to combine multiple weak learners (Bootstrap), to come up with a strong learner (Aggregation), wherein the evaluation metrics of the classifiers are also increased. Bagging is one such ensemble technique, wherein we apply the bootstrap aggregation technique to each sample and a model is created for each sample & classifier. These models are further combined to test with the test data and the final prediction is made. Now, coming back to the task at hand, we have to apply bagging using the above 3 classifiers. I have used the best_estimators_ of each classifier generated through the SearchCV. So, as suggested, I firstly investigated the performance of the classifiers for the n_estimators in the range of 2 to 20 in steps of 2, and I found out that the scoring metric which I’m using (“f1”) in the parameters (since the accuracy can be skewed), boosted up for the Decision Tree Classifier to 0.65 from 0.59 as seen in the previous question. This was for the {'n_estimators': 18} for my dataset. So, as we increase the n_estimators (ensemble size), we get a good scoring metric.
  </p>
   
   <p align="center" >
  <img src="https://user-images.githubusercontent.com/47216809/127859096-15ab916d-c8fa-484f-aab2-be9563db658f.png"  />
  </p>

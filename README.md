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
          Changing the number of instances (best performing ensemble size) Classification performance: 
</h2>
<p align="justify">
  To start with; Ensemble Learning is an ML technique which is used to combine multiple weak learners (Bootstrap), to come up with a strong learner (Aggregation), wherein the evaluation metrics of the classifiers are also increased. Bagging is one such ensemble technique, wherein we apply the bootstrap aggregation technique to each sample and a model is created for each sample & classifier. These models are further combined to test with the test data and the final prediction is made. Now, coming back to the task at hand, we have to apply bagging using the above 3 classifiers. I have used the best_estimators_ of each classifier generated through the SearchCV. So, as suggested, I firstly investigated the performance of the classifiers for the n_estimators in the range of 2 to 20 in steps of 2, and I found out that the scoring metric which I’m using (“f1”) in the parameters (since the accuracy can be skewed), boosted up for the Decision Tree Classifier to 0.65 from 0.59 as seen in the previous question. This was for the {'n_estimators': 18} for my dataset. So, as we increase the n_estimators (ensemble size), we get a good scoring metric.
  </p>
   
   <p align="center" >
  <img src="https://user-images.githubusercontent.com/47216809/127859096-15ab916d-c8fa-484f-aab2-be9563db658f.png"  />
  </p>
  
  <p align="justify">
  As we can see from the above table, these are the evaluation measures after increasing the ensemble size of the bagging classifier from 2 to 20 in steps of 2. I found out that heuristically, 16, 18 and sometimes 20 n_estimators gave improved results than 1. a. classifier models. But as I have chosen “f1” as my scoring measure, we see here that the Decision Tree has outperformed the other classifiers. And this is what I was expecting. Now, I kept the “n_estimators = 18” and changed the max_samples (bag size) parameter from 0.0 to 1.0 and found out that the scores are increasing as we increase the bag size. They are NaN for 0.0 and for anything above than 1.0. However, for a bag size of 1.0 they are giving high metrics, which is already kept default by sklearn.BaggingClassifier() library. But for the 1-NN, for the max_samples of 0.4-0.6, it was giving a good f1 score. Whereas, more the bag size, more was the metric scores for the MLP (Neural Network) and Decision Tree Classifier. Hence, until now the best performing parameters from which we got the above scores as evident in the table are:
  </p>
  
  <ul>
<li> Decision Tree Classifier = BaggingClassifier(dtc, n_estimators=18, max_samples = 0.8, random_state=0)
         </li>
         <li> Neural Network = BaggingClassifier(mlp, n_estimators = 18, max_samples=0.6, random_state=0)
         </li>
         <li> 1-NN Classifier = BaggingClassifier(knn1, n_estimators = 18, max_samples=0.8, random_state=0)
         </li>

</ul>

<p align="justify">
 Herein, I again used RandomizedSearchCV to find these best parameters, but as suggested I first found out the best value for n_estimators, and further kept it constant to find the max_samples (bag size). Hence, we can sufficely rely on these parameters, since the SearchCV hyperparameter tuning is reliable and estimates the best params in less time. We can also see that the results (evaluation measures) from these parameters are better than the normal classifier models (without ensembles). By performing this I was particularly able to differentiate what the parameters n_estimators, max_samples mean and how we can use the hyperparameter tuning in all 3 circumstances (classifiers) to find the best values of these.
  </p>

<h2>
          Changing the ensemble size (random subspacing) Classification performance: 
</h2>
<p align="justify">
  Herein, again we use the BaggingClassifer() in python, and we will tweak the max_features parameter to find the best subspace size at which the classifier models are performing better. We’ve already got the ensemble size from the previous question, which I’ll use to find the best_params_ for max_features (subspace size). So, first we’ll have a look at the results, and further I’ll elaborate them:
  </p>
   
   <p align="center" >
  <img src="https://user-images.githubusercontent.com/47216809/127879150-aaeaa296-1355-4c63-9fc0-59a1413402a5.png"  />
  </p>
  
 <p align="justify">
  As evident from the table above, it is shocking to see an accuracy and f1 score of 1-nn model with just 0.5 max_features, this is because the feature subset is reduced and the samples are not similar to each other, so we have more diversity and we get a good improved result for the 1-nn classifier. For the Decision tree and Neural network, we do get an increased f1 score and accuracy, but they need high number of max_features to find best possible results. In fact, for the Decision Tree Classifier, I was not getting an f1 score until max_features =0.9, and hence it was the best parameter for the decision tree. For Neural Network, it was that, as we increase the max_features from 0.1 to 1.0, the scoring (“f1”) score was increasing and for 1.0 it was the highest.
  </p> 
  
  <p align="justify">
 Hence, we can say that the random sub spacing does affect the classification performance of the chosen model classifiers as stated above and as evident from the table. However, I felt that the f1 score is too low, which is agreed, since I have not used SMOTE or any other class sampling technique to balance the class distribution. In fact, I wanted it to keep it as it is and train the classifier on it with 35% as test set, and I have achievably good results, ignoring the skewed class distribution. As I said earlier, I wanted to see how Random Forest Classifier will act after Bagging, so I ran it with the best params again using RandomizedSearchCV() and got unexpected results. The best params were: BaggingClassifier(base_estimator=RandomForestClassifier(max_depth=13, max_features='log2', n_estimators=379, n_jobs=-1, random_state=0),  _features=0.9, max_samples=0.5, n_estimators=42, random_state=0).
  </p> 
  
  <p align="justify">
  The results were: f1: 0.679, accuracy: 0.869, balanced_accuracy: 0.775, precision: 0.777, recall: 0.600, roc_auc score: 0.792. The results as we can see are close to the Decision Tree Classifier, it means that there might be some another better classifier for my census dataset, which can do much better than this. Or it might be due to a less splitted training dataset. The model classifier might need more data to train. And to make it clear again, all these scores are test_scores, as in the evaluation measures generated after predicting the X_test. So, all in all, considering that I have not used SMOTE and the somewhat imbalanced class set, I’m content with the results.
  </p> 
  
  <p align="justify">
   I also went ahead and implemented AdaBoost, and found out that the f1 score is still around 0.60 for Decision Tree, Random Forest and Neural Network too. Even for 1-nn the f1 score was 0.59. Which means that the model is not overfitting and we are getting proper evaluation metric values. Bagging, however has possibly able to reduce the variance, which was evident from the best_score_ and cv_results_ parameters of SearchCV, as I used the BaggingClassifier with SearchCV.
  </p> 

 
 <h2>
            Best Ensemble Strategy:
</h2>
<p align="justify">
  As seen in the tutorials (practically) and the lectures (theoretically), Bagging benefits the Decision Tree and Random Forest Classifier. Neural Networks are also improved in Bagging techniques. The Random subs pacing benefits the 1-NN Classifier, since it encourages diversity in the method and k-NN works best here. The main difference between Bagging and Random Sub spacing is that the samples are chosen with replacement and without replacement from the Bagging and Random Sub spacing training sets respectively. 
  </p>
  
  <p align="justify">
  As the ensemble methods ensures reliability in the evaluation measures taking efficiency, stability and robustness into consideration, they are trustable. Hence, I’ll say that the things which we have seen in lectures are in accordance with the results which I got for Bagging, Random sub spacing and even for the normal models. As I did this from scratch in python, I’m more than content with the results.
  </p>
  
   <p align="justify">
  With that said, the best ensemble strategies according to my dataset for the said model classifiers are:
  </p>
  
  <p align="center" >
  <img src="https://user-images.githubusercontent.com/47216809/127879653-4e02ddc0-1735-43fa-83db-463d308494f0.png"  />
  </p>

 <p align="justify">
  Now, why so? Well! Decision trees are unstable and tend to overfit, bagging works like wonders by reducing the variance and also clears the notion of over-fitting if any. Same is for the Neural Networks (Multi-Layer Perceptron Classifier), they are also unstable and tend to overfit due to a large number of hidden layers. Bootstrap aggregation (with replacement) avoids this and improves the performance of the NN drastically. However, for my dataset, it was not that drastic, for either the Decision Tree or the Neural Network. Even for k-NN, Bagging (with replacement) is not that effective.
  </p>
  
  <p align="justify">
  However, Random sub spacing is a technique which works on the max_features parameter of BaggingClassifier() in Python, and the 1-NN (k-NN) classifier’s performance drastically improved as seen in the result table in Q.1.c. As said, this is because of the diversity in the ensemble, which works amazingly well for a distance measured model classifier algorithm like K Nearest Neighbour. This was totally as expected; however, I was really not expecting 1-NN to perform so much better after tweaking the max_features parameter (subspace size) using the SearchCV.
  </p>
  
  <p align="justify">
 In the end, I also tried the sklearn’s VotingClassifier, wherein I used all the given models with best_params as the estimators’ parameter along with the ‘soft’ voting parameter, and the f1 score was 0.64 with a balanced accuracy of 0.76, which was a satisfiable evaluation metric. I also used the trending and chartbusting algorithms on Kaggle – XGBOOST & CATBOOST; and I was amazed to see some enhancing results. Previously we were getting the ROC-AUC score around the 70s, but in XGBOOST the ROC-AUC score up hilled to 0.88 and the Type I and Type II errors were also below than 15%, which was quite impressive. But, when I ran the non-hyper tuned CATBOOST, the F1 score boosted to around 68%, while reducing the Type I and Type II errors to below than 10%, which was amazing. I can just imagine, what a properly tuned CATBOOST might be capable of!
  </p>
  
  <p align="justify">
 To conclude, I repeated the whole process by using “SMOTE” in python, but there was a lot of bias and variance for a single class distribution for the Decision Tree and Neural Network model, in some cases the FP’s were totally 0 and in some cases the FN’s were abundant, and vice versa. Hence, I was right with the decision of not using SMOTE in the initial stage of the assignment, and therefore going with evaluation measures like F1 score, balanced accuracy and others as discussed in the previous questions. 
  </p>
  
  
   <p align="center"> <i>   “Accuracy is flawed in many circumstances. I see many Kaggle notebooks which are on the top lists and are using accuracy as the best evaluation measure to conclude their analysis, well! I feel that the aim in any Machine Learning Evaluation should be to reduce the Type I and Type II errors to as low as possible.” </i> </p> </p>

##

<p align="justify"> <b>Question: </b> Comment on the interpretability of different supervised learning techniques. How
easy or difficult it is to explain the reason behind predictions to a layman? Can
you easily find out which training examples need to be modified to change the
prediction for a particular query? Can you easily find out the weight of the
different features in your model? </p>

<p align="justify"> <b>Answer: </b> Interpretability is an essential part of comprehending the various supervised learning, or any Machine Learning techniques. Even though you have the desired results/outputs, it is a difficult task to elaborate the same work to the end-user. I feel that interpreting supervised learning techniques are much simpler than unsupervised learning algorithms. It is because, basic supervised learning algorithms like Decision Trees, Naïve Bayes, Nearest Neighbor, and many more have a predefined dataset and the task is to basically model the dataset between the independent and dependent variables. Though, in layman’s terms we can say that supervised machine learning algorithm is a technique with which computers are able to learn from the given input data which is already labeled and give out expected classification results based on the output results. All in all, in reality, it is quite difficult to explain the intuition behind the predictions made by the machine learning algorithms, since many heuristics, mathematical models and evaluation measures are used by the ML algo. to compute the predictions. So, basically, we, without a machine learning algorithm, cannot “easily” modify the training examples to change the prediction according to one’s expectation query. Fundamental supervised Learning algorithms like Decision Trees, KNN and Naïve Bayes have different strategies in finding out the classification labels, and thus they have different ways to figure out which training examples need to be modified to change the prediction of a particular query. As Decision Trees are rule-based classifications; the Entropy and Gini impurity are some factors along with their hyperparameter tuning, which might be helpful in finding out which training examples that are needed to be modified. On the other hand, KNN and Naïve Bayes are totally opposite, wherein, one is a distance based lazy learner and other is a contingency based, conditionally independent – eager learner. So, in essence, if we know the mathematical reasoning and the crux of these algorithms then one can easily find out which training examples are needed to be modified. Talking about finding weight of various features in the model, is also a very subjective topic, wherein the concepts like information gain, feature selection techniques like filter, wrapper, selective based, exhaustive based, or some statistical tests like Quasi-constant features, Correlation matrix, Annova test, chi-squared scoring test, ROC-AUC scoring test and what not, exists, even after which we are just relying on the produced results. The way we chose the features really matters to some supervised algorithms like Decision Trees, wherein we use the mutual_info_classif model selection technique to find the weights of the features and select the best k features for fitting the model. So, how we select the features matters the most to the Decision Trees algorithm, while in a distance-based algorithm like KNN, the features don’t matter, one thing which matters is that the numerical values should be in a fixed range to get find the best classifications based on the k nearest neighbors, and that’s why we do feature scaling, whereas in the rule-based algorithms, feature scaling doesn’t make any difference. However, I feel there is no need to give weights to features if the task has not specifically mentioned to do so. And to conclude, the supervised machine learning algorithms are much more interpretable rather than the black boxed neural, deep networks which subside by the reinforcement nature of the advanced machine learning and hence making them uninterpretable for a layman or any human to really comprehend what is going on in there! </p>

##

<p align="justify"> <b>Question: </b> Explain the bias-variance tradeoff. Which classifiers generally suffer from high bias
and which classifiers generally suffer from high variance? Which ensemble
strategy helps you to deal with bias and which ensemble strategy helps you to
deal with variance issues? </p>

<p align="justify"> <b>Answer: </b> Before talking about the trade-off, first let’s have a look at what Bias and Variance is. So, bias is basically what the model has predicted (Predicted value) and what is exactly the correct prediction which we are expecting (Expected value). The variance is basically the extra noise related to the training data. Variance is the cause of overfitting, since models with high variance perform well on training data, but have significant drop in the metrics on testing data. Now, talking about Bias-Variance trade-off, it exists to avoid both overfitting (High variance) and underfitting (High bias). So, if either the bias or variance is bigger, there exists an issue with the fitted model. The classifiers which suffer from High bias are Naïve Bayes, KNN, and sometimes logistic regression too. The value of K should be optimally chosen using either SearchCV or some other technique, since a low k value can cause high variance and a very high k value can cause high bias. So, to optimize the value of K, we need to balance the bias-variance trade-off error. Classifiers such as Decision Trees, are tend to have high variance, if the tree is too bushy. So, we can use the max_depth parameter tuning to basically set the depth of the decision tree model, in short; it can be pruned to reduce the variance. With that said, the decision tree makes no assumptions about its results, since it is purely calculated based on the entropy or gini criterion, but its dependent variable is prone to have variance. To tackle such issues of the base models (weak learners), we use ensemble strategies to basically aggregate them further and reduce the said bias/variance to come up with a better model. So, talking about which ensemble strategy deals with bias and variance is fairly explanatory. The Bagging ensemble technique bags in different weak learners and learns from them independently in a parallel fashion (Bootstrapping) and comes up with a deterministic model (Aggregation), which mainly focuses on the variance issue of the base model. Boosting on the other hand, boosts the base models sequentially and forms an adaptive model which focuses on the bias part of the base model. Boosting also focuses on the variance part, but it is prone to noise because of its sequential pattern, which may lead to a high variance (overfitting). However, I feel Bagging is more robust and reliable than Boosting. </p>

##

<p align="justify"> <b>Question: </b> What are the main limitations of k-means in general and Lloyd's algorithm in
particular? Explain how the centroids are initialised in the k-means++ algorithm?
What is the intuition for initialising the centroids in this way? </p>

<p align="justify"> <b>Answer: </b> Being a centroid-based clustering algorithm, k-means has balancing advantages as well as disadvantages while clustering the datapoints. The biggest limitation of k-means is that we need to choose the value of k manually. It is dependent on the initial values of the centroids, for a better k value it needs good initial centroids to finally club nearest data points in a single cluster. It has severe problems with outliers, since outliers (noise) is also considered to be a part of a cluster in most of the cases, for this DBSCAN could be the best bet, since it handles the outliers efficiently with its epsilon and neighborhood boundary parameters. The K means algorithm deteriorates as the dimensionality of the datapoints increase. For finding the best value of K, we need to loop it until we find the best value, which increases the time complexity in the worst case making it NP-Hard. It forms spherical clusters only, hence, it fails to cluster the datapoints which are not spherical. Now, talking about Lloyd’s algorithm in particular, it is a heuristic method which is used in the k-means clustering algorithm, which converges to local minimum in few steps. Even though it is said to be in the local minima, in actuality it runs numerous times and is an intensive iterative process. Hence, the main drawback of the Lloyd’s algorithm is that it may not converge in polynomial time. The initial findings of the centroids can also be a huge setback for further computation in Lloyd’s algorithm. Llyod’s algorithm can run iteratively until the stopping criteria is not satisfied. As the Llyod’s algorithm uses Euclidean distance, it can only handle numerical data. As the k-means algorithm has these many disadvantages we use k-means++ method for initialization of the centroids. It chooses a center at random among a set of data points, then for the data points which are not chosen, it calculates the distance between them to the nearest center that has been already chosen. We then chose another data point as a new center and do the same thing until k centers have been found. These k centers are your initial centers and we can proceed with the further steps of the k-means algorithm. The fundamental intuition (reason) behind choosing the centroids in this fashion is that the initial center which is chosen at random is basically chosen heuristically such that the next centroid is chosen from the left-over data points with the probability directly equivalent to the squared distance from the data point’s closest existing cluster centroid. Another reason for choosing the centroids in this way. is the approximation ratio, time complexity of O (log k), where k is the number of clusters used. </p>

##

<p align="justify"> <b>Question: </b>What does R2 measure represent in the case of linear regression? </p>

<p align="justify"> <b>Answer: </b> The R2 (R squared) is basically the coefficient of determination of the regression score function. It is a statistical measure which represents the proportion of the variance for the target which is explained by the feature variable/variables in a linear regression model. In layman terms, it basically tells us how much one variables’ variance is influencing (explaining) the other variables’ variance. Hence, if the value is 0.3, then 30% of the observed variation is explained by the feature variables. It is basically given as; 1 – unexplained variance/total variance. If the value is 100%, then both the independent and dependent variables contain the same values, or we can say that the dependent variables variance is successfully (completely) explained by the independent variables. So, in terms of linear regression; we need to find out the r-squared value to see how the model fits the data and to also look how strongly the dependent and independent variables are associated with each other to form a linear relationship between them. </p>

##

<p align="justify"> <b>Question: </b> Explain the precision-recall tradeoff.</p>

<p align="justify"> <b>Answer: </b> Precision-Recall score was used by me in the last assignment where the classes were imbalanced and they were multi-class as well. Precision is basically the evaluation measure of relevancy of the result, whereas the recall is basically how many correctly relevant results are gathered. The tradeoff between these 2 is called as the Precision-Recall trade-off, which is for various thresholds, just like the ROC curve. However, it is much more efficient than the ROC curve since it’s Area under the curve gives us an idea of both a high recall and precision. It is also guaranteeing low FPR and FNR, which are the type errors. But we cannot have both the precision and recall high, as seen in the question 1, where the f1 score was basically around 0.5-0.6 because of a low recall score and high precision score. All in all, we can say that if our priority is precision and not recall or recall and not precision, we can plot the PR-curve. If we need a higher precision value then we set the threshold higher from the default threshold value which is 0, and for a higher recall value then we set the threshold lower than 0. PR Score should be anytime preferred when the class is imbalanced or there is a multi-class classification problem over AUC score. </p>

## 

<p align="justify"> <b>Question: </b> Explain how the parameters are learnt in the training of neural networks.</p>

<p align="justify"> <b>Answer: </b> Neural Network consists of numerous neurons connected to each other, and each connection is associated with a weight and bias. This comprises of the cost function C (weight, bias) to compute the self-contradiction between the predicted features with the real features. This is basically an iterative process by the layers of the neurons, until the best model is attained. The cost function is computed based on the machine learning algorithm which is to be performed. For regression, the cost function which we use is the mean squared error while for classification, we use cross entropy. The parameters for a neural network are chosen by implementing an optimized course of action. The first phase is called as the forward propagation, wherein the network is exposed to the training data where the prediction labels are calculated, after which the loss (cost) function is used to compare how the prediction result was in relation to the real result, which should be 0, since we want no divergence between the predicted and real results. We then perform backward propagation in order to propagate the loss to all the parameters consisting the Neural Networks. We then use this information to update the parameters with the gradient descent, wherein the total cost is reduced and we finally obtain a better model. Gradient Descent is an iterative optimization algorithm and it helps us to get closer to the local minimum. But rather than using a batch gradient descent, in order to learn the parameters for a neural network, we use the stochastic gradient descent to enhance the computation speed, wherein the gradient is estimated for a scarcely and randomly selected training samples. This also reduces the time to find the total cost of the function, thereby minimizing the overall computation to train the neural network. </p>

## 

 
 <p align="center"> \\\\\\\\\\ THANK - YOU ////////// </p>

          PROJECT CREATED BY - Prashant Wakchaure
          Email ID - prashant900555@gmail.com
          Contact No. - +373 892276183


  

  

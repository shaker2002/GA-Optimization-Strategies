## Overview:
The project target is to compare the different optimization strategies for the genetic algorithm in the feature selection problem and see which are the most effective hyper-parameters for this algorithm in this task. <br>
The main representation of the problem is a bit-string representation where 1 indicates taking the feature and 0 indicates not taking it. In the genetic algorithm implementation, many techniques were used, such as testing the data on multiple classification algorithms, trying different parent selection techniques, survival selection techniques, cross-over methods, and applying some mutation. All of those will be discussed in detail in the next few pages.<br>
The project was created in a way that works on any classification task dataset, such that the target class is the last column in the dataset.<br>
Finally, the results of all the techniques are saved as a CSV file so that they can be compared to conclude the best summary.

## The datasets:

1-	The diabetes dataset:<br>
The Pima Indians Diabetes dataset contains medical records of female patients of Pima Indian heritage, aged 21 and older, aimed at predicting diabetes occurrence. It includes diagnostic health metrics and a binary outcome indicating diabetes status. It has 8 features and a binary classification class “Outcome”.<br>
Link: https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database

2-	Breast Cancer dataset:<br>
The Breast Cancer dataset contains medical records with attributes used to classify breast cancer cases as malignant or benign. It includes features derived from imaging data, such as tumor size and texture, and is designed for binary classification tasks to support cancer diagnosis.
It has 31 features and a binary classification class “diagnosis”.<br>
Link: https://www.kaggle.com/datasets/rahmasleam/breast-cancer?resource=download

3-	Predict Students' Dropout and Academic Success dataset(data2):<br>
This dataset analyzes student dropout and academic success using features derived from demographic, academic, and socioeconomic data for 4,420 students. It is ideal for building classification models to predict student outcomes and identify factors influencing educational success.
It has 36 features and a multiclass classification class “Target”.<br>
Link: https://archive.ics.uci.edu/dataset/697/predict+students+dropout+and+academic+success


## Code overview:
### 1-	Survival selection techniques:<br>
Every generation an offspring is generated that will replace the population using one of these techniques: <br>
* a.	Elitism:
 The top 10% of the population is carried over to the offspring (depending on the accuracy).
* b.	Genitor:
 The worst 10% of the population is removed and the rest is carried over to the offspring (depending on the accuracy) “It was practically implemented by carrying the top 90% of the population to the offspring”.
* c.	M, l:
 Nothing is carried from the population to the offspring.
* d.	M + l:
The top 50% of the population is carried to the offspring (depending on the accuracy).

### 2-	Parent selection techniques:<br>
A parent_selection function returns 2 solutions from the population as parents for either crossover or mutation using one of these techniques:<br>
* a.	Uniform:
All the individuals have the same probability of being selected.
* b.	Fitness Proportionate Selection (fps):
The probability of the individual to be selected is proportionate to his accuracy performance.

### 3-	Crossover techniques:<br>
A crossover function takes 2 parents and returns a single child using one of these techniques:<br>
* a.	1-point:
Select a random point in the range of the length of the parents and cross over that point.
* b.	N-point:
Select n random points in the range of the length of the parents and cross over these points.
* c.	Uniform:
Takes the first gene from the first parent and the last gene from the second parent and randomizes the rest of the child.

### 4-	Mutation:<br>
A mutation function takes a single parent and applies a bit-flipping mutation to it on a random number of genes and returns the child.

### 5-	Population evaluation:<br>
An evaluation function takes a data set and splits it into training and testing data and then using the population it determines which features are going to be used in the evaluation using one of those classification algorithms and returns the accuracies:<br>
* a.	KNN.
* b.	Decision tree(dt).

### 6-	Running methods:<br>
There are 2 functions that can run the model to fine-tune it using all the previous techniques to improve the population:<br>
* a.	Run until a threshold is met:
An accuracy threshold is declared and the function keeps applying the population manipulation techniques to improve the population’s accuracy until the accuracy meets the threshold.
* b.	Run a specific number of generations: 
The function keeps applying the population manipulation techniques for the given number of iterations/generations.

### 7-	Result saving:<br>
To be able to compare all the possible results, multiple loops were made that loop over every survival selection technique, parent selection technique, Crossover technique, and classification algorithm.<br>
Every iteration, a list keeps track of the evaluation algorithm used, survival selection technique, parent selection technique, crossover method, the number of generations, accuracy, and the number of features. That list gets appended to a data frame, and at the end, the data frame gets converted to a CSV file and saved.


## Results conclusion:
All the tests used the “run until threshold reached” method, each data set had a different threshold depending on the average accuracy of each data set, and all of the tests were on a population size of 10 individuals. The mutation probability is 0.1 and the cross-over probability is 0.5, the results were as follows:


### 1-	The diabetes dataset:

For this dataset, the threshold was 0.74.<br>
The first thing noted is that the KNN algorithm reduces the number of features significantly compared to the decision tree, where the KNN needs between (3-5) features while the decision tree needs between (6-8) features, The second thing noted is that the decision tree algorithm gave better accuracies reaching 0.78 compared to the KNN which reached 0.746.<br>
So, the conclusion is that when using the decision tree algorithm, it is better to use the “m + l” survival selection method, “uniform” parent selection method, and “uniform” crossover, but when using the KNN algorithm it is better to use the “m, l” survival selection method, “fps” parent selection, and “1-point” crossover.<br>
But overall, it is recommended to use the decision tree algorithm for this dataset which gave this outcome: [1,1,0,1,1,1,1,1]


### 2-	Breast Cancer dataset:

for this dataset, the threshold was 0.98.<br>
the first thing noted is that the KNN algorithm needs on average way fewer generations to reach the threshold compared to the dissection tree algorithm, The second thing noted is that again the dissection tree gave better accuracies reaching 0.991 compared to the KNN algorithm which reached 0.982.<br>
So, the conclusion is that if the main target is to get the highest accuracy, then the decision tree algorithm with “m, l” survival selection method, “uniform” parent selection method, and “uniform” crossover is the best choice with a 0.991 accuracy and 12 features which are: [0,0,0,1,1,0,0,0,0,1,1,1,0,0,1,0,0,0,1,1,0,0,1,0,1,0,0,0,1,1], But if the main target was to have the least number of features, Then the KNN algorithm with “genitor” survival selection method, “fps” parent selection, and “uniform” crossover is the best choice with 0.982 accuracy and 8 features which are : [0,0,0,1,0,0,0,0,0,1,0,0,1,0,0,0,0,1,0,0,1,0,1,0,0,1,0,0,0,1].

### 3-	Predict Students' Dropout and Academic Success dataset(data2):

for this dataset, the threshold was 0.7.<br>
the first thing noted is that the decision tree algorithm on average outperformed the KNN, either on the accuracy side or on the number of features side the decision tree had higher accuracy reaching 0.729, and a smaller number of features reaching 8 features.<br>
So, the conclusion is that it is better to use the decision tree algorithm on this data set with an “elitism” survival selection method, “uniform” parent selection, and “n-point” crossover which will give an accuracy of 0.718 and 8 features which are:<br>
 [0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,1,0,1,0,1,0,0,1,0,0,0,0,1,0,0,1,0,0,1,0,0].


## The results for every dataset is in the result csv file in the repo

### I am proud to say that I did not use AI for this project. I did everything without even an internet connection.




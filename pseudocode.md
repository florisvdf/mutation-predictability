## Pseudocode of the analysis
This file describes the steps implemented in our notebooks to generate the main findings 
reported in our article. 

### Variables

`structural classes` = The eight structural classes described in the article (buried, 
exposed, many contacts, few contacts, etc.)\
`data` = The complete dataset of variants sequences and accompanying properties to 
predict\
`model` = Any of the four machine learning models described in the article.


### Functions
`get_data_by_structural_class` = Returns a subset of the dataset that corresponds to 
variants that belong to the passed structural class.\
`k_fold_cross_validation` = Splits the data into k different folds, where each fold 
represents a partition of the dataset. It returns 10 different sets of train and test 
indices, allowing for model evaluation through cross-validation.\
`train_model` = Trains the specified machine learning model on the provided training 
data, using the features and labels in the training dataset. \
`predict` = Takes a trained machine learning model and a dataset as input and generates 
predictions for the target variable using the model. It returns the predicted values 
based on the input features. This function is used to apply the trained model to test 
data.\
`evaluate` = Evaluates the spearman correlation of a model by comparing its predictions 
to the actual values in a test dataset.\
`aggregate_results` = Computes summary statistics of the test performance scores over 
all 10 folds of models for all structural classes of variants.



### Pseudocode

```angular2html
performances = {}
for class in structural_classes:
    group = get_data_by_structural_class(data, class)
    for (train_indices, test_indices) in k_fold_cross_validation(group, 10):
        train_group = group[train_indices]
        test_group = group[test_indices]
        model = train_model(model, train_group)
        test_predictions = predict(model, test_group)
        performances[class].append(evaluate(test_predictions, test_group))
results = aggregate_results(performances)
    
```
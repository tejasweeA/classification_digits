Output of Confusion Matrix:

Train Data splits:

1.	10% of 80% train dataset			2. 20% of 80% train dataset
      

3.	30% of 80% train dataset			4. 40% of 80% train dataset

    


5. 50% of 80% train dataset			6. 60% of 80% train dataset                  


7. 70% of 80% train dataset			8. 80% of 80% train dataset  

9. 90% of 80% train dataset			10. 100% of 80% train dataset                    


Graph of Various Train Splits vs F1 score of Test score:

 


Output obtained on the terminal:

Below given snippet shows the metrics of f1 score and accuracy of test dataset, respective gamma values, train split (in terms of percentage) and f1 score and accuracy of validation dataset.

 

Conclusion:

As per the observation drawn from the graph and the confusion matrices of different training ratios of 80% of the dataset, it can be seen that efficient results are given by 70% of the given training dataset.
This is because it does not only give maximum f1 score on the test dataset with relatively lesser ratio of the training set but also, the confusion matrix has all zeroes except the principal diagonal elements.

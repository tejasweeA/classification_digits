Output of Confusion Matrix:

Train Data splits:

1.	10% of 80% train dataset			
      ![image](https://user-images.githubusercontent.com/89181401/143923455-9c5a8760-a2aa-414d-9dc5-16e95bb464b5.png) 

2. 20% of 80% train dataset

![image](https://user-images.githubusercontent.com/89181401/143923517-cf008421-900b-4d63-bb97-37bdca11b44a.png)

3.	30% of 80% train dataset

![image](https://user-images.githubusercontent.com/89181401/143923728-eabeb6ec-8f8a-45df-bfdc-dea654f1d104.png)

4.	40% of 80% train dataset
![image](https://user-images.githubusercontent.com/89181401/143923757-c77ff914-adf9-4ad6-bfeb-4ef02c7e56f1.png)

5.	50% of 80% train dataset
![image](https://user-images.githubusercontent.com/89181401/143923807-541a8da5-8937-45ae-a7c2-6fb79eb5b87a.png)

6.	60% of 80% train dataset
![image](https://user-images.githubusercontent.com/89181401/143923832-f08141ac-420b-41d7-8c6d-cb490ef2ae1a.png)

7.	70% of 80% train dataset
![image](https://user-images.githubusercontent.com/89181401/143923891-7cc520f3-968e-4ec4-8b31-f97ceb4694cb.png)

8.	80% of 80% train dataset
![image](https://user-images.githubusercontent.com/89181401/143923923-8264a505-01d6-4d0c-988e-3dbcab6baa16.png)

9.	90% of 80% train dataset
![image](https://user-images.githubusercontent.com/89181401/143923950-4a3a0070-7655-4e94-b63f-fd1dfae6371f.png)

10.	100% of 80% train dataset
![image](https://user-images.githubusercontent.com/89181401/143923975-0445040d-e971-4553-b7e8-5b3f787d3ca8.png)
                    


Graph of Various Train Splits vs F1 score of Test score:

 ![image](https://user-images.githubusercontent.com/89181401/143923989-f2f7189d-72d2-4f9d-8de1-60c4cd006dfa.png)



Output obtained on the terminal:

Below given snippet shows the metrics of f1 score and accuracy of test dataset, respective gamma values, train split (in terms of percentage) and f1 score and accuracy of validation dataset.

![image](https://user-images.githubusercontent.com/89181401/144608853-d7253dc1-090c-46df-ba09-40d505c35149.png)


Conclusion:

As per the observation drawn from the graph and the confusion matrices of different training ratios of 80% of the dataset, it can be seen that efficient results are given by 70% of the given training dataset.
This is because it does not only give maximum f1 score on the test dataset with relatively lesser ratio of the training set but also, the confusion matrix has all zeroes except the principal diagonal elements.

![image](https://user-images.githubusercontent.com/12669848/189498908-09206c0c-5e48-4164-8f4c-edccb593cd19.png)



# Predicting Tags for StackOverflow Questions

## Summary

Stack Overflow is a community based Q&A website and is the largest online community for programmers to learn, share their knowledge, and advance their careers. It has been the single source of truth to answer programming questions for even the best of programmers. Along with having an active community that eagerly helps in providing answers to new questions, the sheer volume of of those questions asked is so large that it is impossible to prevent duplicate questions. Therefore, StackOverflow has come up with multiple solutions to prevent question replication. One of the solutions is to have a downvote button and another one of the solutions is to "TAG" the questions so that community members can search for the appropriate questions before having to ask the same question again. It does this using multiple methods like Text Processing, Neural Networks and various Machine Learning algorithms. This project is an humble effort to duplicate the "TAG" feature for StackOverflow questions. 

## Data Source

### Original Dataset 

The data was sourced from Kaggle and is part of a larger StackOverflow dataset that was extracted using Google's BigQuery data warehousing tool. The original dataset includes an archive of StackOverflow content, including posts, votes, tags, and badges. The dataset is updated to mirror the StackOverflow content on the Internet Archive, and is also available through the Stack Exchange Data Explorer. 

### Data Sampled

The dataset was downloaded from : https://www.kaggle.com/datasets/stackoverflow/stackoverflow

`The Python Notebook was created in Google Colab and since it provides a seamless integration with Google Drive, the datasets were manually downloaded and uploaded on Google Drive.`

The data that I used contains text from 10% of questions and tags from the original dataset. Since the main goal of this project is to create a model that given a question text predicts the tags, other data like votes, tags, badges have been removed. We have the data in form of two .csv files:  

1. **Questions.csv** : contains the title, body, creation date, closed date (if applicable), score, and owner ID for all non-deleted Stack Overflow questions whose Id is a multiple of 10.

![image](https://user-images.githubusercontent.com/12669848/189503565-cc036bde-52fc-4cf5-a501-25d69ffccc2e.png)

2. **Tags.csv** : contains the tags on each of these questions

![image](https://user-images.githubusercontent.com/12669848/189503585-12abc60b-9de2-40a3-a7ab-2c2f84756400.png)

Please refer to the diagram below to understand the data. Upon initial analysis, it becomes evitable that it would be mandatory to merge the two datasets for model prediction. Therefore, in the next step, we start with merging, cleaning and truly understanding the data. 

![image](https://user-images.githubusercontent.com/12669848/189506215-fc5d762c-37d3-4045-aa10-37ec3e1f0aff.png)


## Data Cleaning

Before, we move on to the EDA, there were a bunch of steps performed in order to clean the data. The first steps were to analyze if there were any null / na values, changing dtype objects with the correct data types (mostly String) 

### Merge Data
1. In order to merge the data, I grouped the tags by the Id of the question since a question can have multiple tags. I then use the groupby function to merge the dataframes on the `Id`. The Tags were stored as a space-separated string of Tags for every question. 
2. I separated the list into a list of individual tag strings. This makes it into a multi-label classification problem where 1 question can have multiple Tags associated with it. 

*For example: *



### Duplicates
1. Used the pandas `duplicates()` function to find if there are any duplicates in the data.

### Drop Additional attributes
1. 

## Exploratory Data Analysis

## Data Preparation

## Machine Learning Models

## Optimize Model parameters

## Accuracy Metrics




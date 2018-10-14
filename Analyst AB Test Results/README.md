
# Analyze A/B Test Results

## Summary

For this project, we will be working to understand the results of an A/B test run by an e-commerce website. Our goal is to work through this notebook to help the company understand if they should implement the new page, keep the old page, or perhaps run the experiment longer to make their decision.

## Prerequistes

You will need : Pandas, numpy, random, matplotlib.pyplot, and statsmodels.api

some CSV files : ab_data.csv, countries.csv


## Project Steps & Goals

Part I - Probability :

Read the dataset, find the unique users, proportion of users converted, mismatch lines, and missing value.

Part II - A/B Test :

1. Identify the null and alternative hypotheses
2. Compute the probabilities of converting for old and new pages and different groups
3. Perform hypothesis testing and find the p-values

Part III - A regression approach :

1. Compute logistic regression
2. Find dummy varaiables 
3. find the p-value and inspect result if it match with previous result

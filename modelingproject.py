# -*- coding: utf-8 -*-
"""
Created on Sun Apr 11 11:49:31 2021

@author: mmalett
"""

# Modeling Project
# T tests 

# Experiments 1 and 2 old and young people 
import pandas as pd
from scipy import stats 

# Part 1: Read in the data from excel
df = pd.read_csv(r'C:\Users\Melissa\Documents\bme6311\modelingproject.csv')

# Part 2: Read column data of means 
old_MIS=df["old mis"]
old_open=df["old open"]
young_MIS=df["young mis"]
young_open=df["young open"]

# Part 3: old people t-test
t1,p1=stats.ttest_ind(old_open,old_MIS)
print("t = " + str(t1))
print("p = " + str(p1))

# Part 4: young people t-test
t2,p2=stats.ttest_ind(young_open,young_MIS)
print("t = " + str(t2))
print("p = " + str(p2))

# Part 5: Compare old and young MIS 
t3,p3=stats.ttest_ind(old_MIS,young_MIS)
print("t = " + str(t3))
print("p = " + str(p3))
#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


## importing the datasets and storing them in a single dataframe
df = pd.read_csv('assessments1.csv', "|")
for i in range(2,5):
    df.append(pd.read_csv(f'assessments{i}.csv', "|"))


# # Part 1: Analysis

# # Q2. Most important analysis for the product-owner

# In[3]:


## getting a general idea of the dataset
df.describe()


# In[4]:


## analysis for the product owner--something is wrong, scores should either be 0 or 1
df[df.item_score<0]


# In[5]:


## analysis for the product owner--something is wrong, time (in secs) can never be negative
df[df.time_in_seconds<0]


# In[6]:


def indicator(data):
    if (data.time_in_seconds<0) or (data.item_score<0):
        return True
    else: return False

df['error'] = df.apply(indicator, axis=1)


# In[7]:


df.error.value_counts().plot(kind='pie', autopct="%1.1f%%")


# In[8]:


df.error.value_counts()


# ## Analysis for the product owner
# 1. About 0.6% of the data is inaccurate/Assessment data for 1,400+ students affected
# 2. 700+ records with negative time (should always be positive)
# 3. 700+ records with negative score (should be either 0 or -1)
# 
# ### Advice: Please check the data collection process/logic--it seems that the data is not being recorded accurately and about 0.6% records are affected by it

# # Q1. Analysis for the CEO

# In[9]:


## filtering out the required data and filling NA values with a boolean value denoting that the question was not attempted
df = df[(df.item_score>=0) & (df.time_in_seconds>=0)]
df['attempted'].fillna(False, inplace=True)
df


# In[10]:


print(f'We are currently active in {len(df.district_id.unique())} districts and {len(df.school_id.unique())} schools')


# In[11]:


## printing values for total number of unique students and assessments
print(len(set(df.school_id) | set(df.student_id)), len(df.assessment_id.unique()))


# In[12]:


## filtering the required colummns to work on--prepping for data summarization for further analysis
## NOTE: item_id is summarized as count--denoting the total count of questions in an assessment session
working = df[['school_id', 'student_id', 'assessment_id', 'grade', 'item_id', 'item_score']]
wp = working.pivot_table(index=['assessment_id'], values=['item_id', 'item_score'], aggfunc={'item_id':'count', 'item_score':'sum'})


# In[13]:


## computing required columns
wp.reset_index(inplace=True)
wp['percentage'] = wp.item_score/wp.item_id


# In[14]:


wp.sort_values('percentage')


# In[15]:


## bucketing the data in terms of percentage
wp['category'] = np.where(wp.percentage>=0.5, 'Above 50%', 'Below 50%')
wp


# In[16]:


## plotting a pie-chart
pie = wp.category.value_counts().plot(kind='pie', autopct="%1.1f%%")
pie.set_title('Percentage of assessments with less than 50% scores')


# ## Analysis for the CEO
# * We are active in 15 districts, 300+ schools (with 5600+ active students)
# * On an average, a student participated in 0.78 assessments
# * Students scored 50% or higher in about 80% of the assessments--denoting that the difficulty is leaning towards the 'easy' side of the spectrum
# 
# ### Advice: Need to work on engagement--the minimum criteria is to have an engagement value of 1 (#students = #assessments); plan for students not participating. Difficulty needs to be adjusted, seems to be a little easy for students since about 80% of the assessment results are 50% or higher--probably time to implement competitive/adaptive logic to adapt to individual student's learning requirement.

# In[17]:


df[(df.assessment_id=='assessment_634') & (df.student_id=='student_3212')]


# In[18]:


len(df.student_id.unique())


# # Q3. Analysis for the educational researcher

# In[19]:


student_comp = df.pivot_table(index=['school_id', 'student_id', 'assessment_id'], values=['item_id', 'attempted', 'item_score', 'time_in_seconds'],
                aggfunc={'item_id':'count', 'attempted':'sum', 'item_score':'sum', 'time_in_seconds':'sum'}).reset_index()


# In[20]:


## filtering data for records with attempted questions
student_comp = student_comp[student_comp.attempted>0]
student_comp


# In[21]:


## computing required columns
student_comp['percentage'] = student_comp['item_score']/student_comp['item_id']
student_comp['attempt_percent'] = student_comp['attempted']/student_comp['item_id']
student_comp['average_time_pq'] = student_comp['time_in_seconds']/student_comp['item_id']
student_comp.sort_values(['percentage', 'attempt_percent', 'time_in_seconds'], ascending=[False, False, True], inplace=True)


# In[22]:


student_comp[student_comp.time_in_seconds>=0].describe()


# In[23]:


student_comp[(student_comp['time_in_seconds']>0) & (student_comp['attempt_percent']>0.99)].describe()


# In[24]:


## binning the data in terms of grade (categorized on the basis of percentage score)
## NOTE: Here grade is not the grade in which the student is, instead, it represents the category of the percentage
student_comp.loc[student_comp['percentage'].between(0, 0.25, 'both'), 'grade'] = 'D'
student_comp.loc[student_comp['percentage'].between(0.25, 0.5, 'right'), 'grade'] = 'C'
student_comp.loc[student_comp['percentage'].between(0.5, 0.75, 'right'), 'grade'] = 'B'
student_comp.loc[student_comp['percentage'].between(0.75, 1, 'right'), 'grade'] = 'A'


# In[25]:


## computing median time spent per question for every student in a grade
student_comp[['grade', 'average_time_pq']].groupby('grade').median()


# ## Analysis for the educational researcher
# * A grade-A student spends about 46 seconds (median) in attempting a question
# * It can be seen that students who invest below 39 secs in attempting a question fare poorly
# * A sweet-spot for the students could be investing about 45-60 seconds per question as per difficulty
# * If students over-invest their time, they fare lower than expected (more of a deadlock situation); lower chances to answer the question correctly

# In[26]:


student_comp = student_comp[student_comp.time_in_seconds>0]
student_comp[student_comp.grade=='A'].describe()


# In[27]:


student_comp[student_comp.grade=='B'].describe()


# In[ ]:





# In[28]:


df_geog = df[df.subject=='geography']
df_hcvs = df[(df.subject=='history') | (df.subject=='civics')]


# In[29]:


df_geog


# In[30]:


df.describe()


# In[ ]:





# # Part 2: Hypothesis Testing
# "The head of Learning Sciences has asked you to present evidence to show that performance
# on assessments is related to time spent taking the assessment."

# ## Interpretation
# The head of Learning Sciences has an assumption that the total time invested in an assessment is correlated to the grade/score of a student. A simple hypothesis testing will be sufficient to prove if there is any such evidence.
# Thus, we need to prove if there is any statistical evidence in favour/against of this statement.
# 
# From the interpretation we can conclude that <b><i>our null hypothesis (H0), in this case, would be that the mean time spent of a grade A student would be equal to the mean time spent of a grade D student</i></b>.

# ![image.png](attachment:image.png)

# and consequently, <b><i>our alternate hypothesis (H1) would be that the mean time spent by a grade A student would unequal to the mean time spent by the grade D student in an assessment</i></b>.

# ![image.png](attachment:image.png)

# ## Step 1: Data extraction

# We have already cleaned the required data in part one Q3. analysis for educational researcher, where we looked into the average time spent by a student, per question.
# We shall take the same data, and split it into two datasets, grade_A and grade_D

# In[31]:


grade_D = student_comp[student_comp.grade=='D']
grade_D.describe()


# A sample of 9000 records is then extracted from each dataset, to perform this hypothesis test. Sampling is done to remove some bias in the inherent ordering of the data and helps nullify any statistical correlation caused due to such instances.

# In[32]:


## sampling students from two categories to perform an unpaired t-test
grade_A = student_comp[student_comp.grade=='A'].sample(9000)
grade_D = student_comp[student_comp.grade=='D'].sample(9000)


# ## Step 2: Performing the test

# Since in this situation, we do not know the standard deviation of the population--we only have a sampled dataset at hand, we would have to perform a t-test. Moreover, we would have to perform an unpaired t-test since the samples are taken from two different groups of people participating in the assessment. NOTE: If we knew the sdev. of the population, we would perform a z-test instead.
# 
# ASSUMPTIONS:
# * The sampled datasets would be IID (independent and identical datasets)
# * They would have the same standard deviation
# * Both the datasets would be normally distributed (or of size 30 or higher)
# 
# Lastly, due to our hypothesis formulation, it is clear that <b>we have to perform a two-tailed test considering the level of significance (alpha) for this test to be 0.05 (5% error is acceptable)</b>.

# ![image-2.png](attachment:image-2.png)

# We proceed ahead assuming our sampled datasets satisfy all the requirements mentioned above.

# In[33]:


## importing the required packages and performing the t-test
import scipy.stats as stats
ttest,p_value = stats.ttest_ind(grade_A['time_in_seconds'],grade_D['time_in_seconds'])


# In[34]:


print(ttest, p_value)


# Results during testing:
# ![image.png](attachment:image.png)

# ## Observations/Feature generation
# The test-statistic in this case (t-stat) = 2.70 and the p-value = 0.007
# ### Since p-value is << alpha (level of significance), we have statistical evidence to reject the null hypothesis (H0)
# 
# p-value is a probabilistic value denoting the probability of the test mirroring our null hypothesis, it ranges from 0 to 1 and as p-value approaches towards 1, the probability of our null hypothesis being true increases.

# ## Conclusion

# In[35]:


if p_value<0.05:
    print("Reject null hypothesis (H0)")
else: print("Fail to reject null hypothesis (H0)")
    


# Considering the p-value and level of signicance during testing, we have enough statistical evidence to reject the null hypothesis, we conclude that at the significance level of 0.05, there is correlation of time spent in an assessment and the performance of the student.
# 
# ### The mean time spent by a grade A student and a grade D student in an assessment would not be equal; and thus, performance on assessments is related to time spent taking the assessment.

# ![image-2.png](attachment:image-2.png)

# NOTE:
# if the p-value was greater than or equal to alpha, we would conclude that there is not enough evidence to reject the null hypothesis and thereby, mean time spent by a grade A and a grade D student could be equal.

# In[ ]:





# In[ ]:





# In[36]:


grade_A


# In[37]:


grade_D


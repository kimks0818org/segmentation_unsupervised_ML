# -*- coding: utf-8 -*-
"""
Created on Mon Mar 25 13:18:39 2019

@author: Ki

Purpose: final exam
"""


# Importing libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler  # standard scaler
from sklearn.decomposition import PCA  # principal component analysis
from scipy.cluster.hierarchy import dendrogram, linkage  # dendrograms
from sklearn.cluster import KMeans  # k-means clustering


# Setting pandas print options
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)


# Importing dataset
customers_df = pd.read_excel('finalExam_Mobile_App_Survey_Data.xlsx')


###############################################################################
# Code for Data Analysis: Data Exploration
###############################################################################


# General Summary of the Dataset
print(customers_df.shape)
print(customers_df.info())
print(customers_df.describe().round(2))


customers_desc = customers_df.describe(percentiles=[0.01,
                                                    0.05,
                                                    0.10,
                                                    0.25,
                                                    0.50,
                                                    0.75,
                                                    0.90,
                                                    0.95,
                                                    0.99]).round(2)


customers_desc.loc[['min',
                    '1%',
                    '5%',
                    '10%',
                    '25%',
                    'mean',
                    '50%',
                    '75%',
                    '90%',
                    '95%',
                    '99%',
                    'max'], :]


# Viewing the first few rows of the data
print(customers_df.head(n=5))


# Any missing values?
customers_df.isnull().sum().sum()


########################
# More about the respondents
########################


# Plotting variables regarding age information
fig, ax = plt.subplots(figsize=(12, 8))


plt.subplot(2, 3, 1)
plt.title('Age')
sns.distplot(a=customers_df['q1'],
             hist=True,
             kde=False,
             color='purple')


plt.subplot(2, 3, 2)
plt.title('No Children')
sns.distplot(a=customers_df['q50r1'],
             hist=True,
             kde=False,
             color='green')


plt.subplot(2, 3, 3)
plt.title('Children under 6')
sns.distplot(a=customers_df['q50r2'],
             hist=True,
             kde=False,
             color='yellow')


plt.subplot(2, 3, 4)
plt.title('Children 6-12')
sns.distplot(a=customers_df['q50r3'],
             hist=True,
             kde=False,
             color='red')


plt.subplot(2, 3, 5)
plt.title('Children 13-17')
sns.distplot(a=customers_df['q50r4'],
             hist=True,
             kde=False,
             color='orange')


plt.subplot(2, 3, 6)
plt.title('Children over 18')
sns.distplot(a=customers_df['q50r5'],
             hist=True,
             kde=False,
             color='pink')


plt.tight_layout()
plt.savefig('age_plots.png')
plt.show()


# Plotting remaining demographical information
fig, ax = plt.subplots(figsize=(12, 8))


plt.subplot(2, 3, 1)
plt.title('Education level')
sns.distplot(a=customers_df['q48'],
             hist=True,
             kde=False,
             color='purple')


plt.subplot(2, 3, 2)
plt.title('Marital status (1-Married, 2-Single)')
sns.distplot(a=customers_df['q49'],
             hist=True,
             kde=False,
             color='green')


plt.subplot(2, 3, 3)
plt.title('Race (1-White or Caucasian)')
sns.distplot(a=customers_df['q54'],
             hist=True,
             kde=False,
             color='yellow')


plt.subplot(2, 3, 4)
plt.title('Hispanic or Latino (1-Yes, 2-No)')
sns.distplot(a=customers_df['q55'],
             hist=True,
             kde=False,
             color='red')


plt.subplot(2, 3, 5)
plt.title('Household annual income')
sns.distplot(a=customers_df['q56'],
             hist=True,
             kde=False,
             color='orange')


plt.subplot(2, 3, 6)
plt.title('Gender (1-Male, 2-Female)')
sns.distplot(a=customers_df['q57'],
             hist=True,
             kde=False,
             color='pink')


plt.tight_layout()
plt.savefig('demographic_plots.png')
plt.show()


# Checking class balance for respondent's age variable
customers_df['q1'].value_counts()
sns.distplot(a=customers_df['q1'])
plt.show()
plt.clf()


# Set dictionary to merge age classes (Generalize)
age_group = {1: 10,
             2: 20,
             3: 20,
             4: 30,
             5: 30,
             6: 40,
             7: 40,
             8: 50,
             9: 50,
             10: 60,
             11: 60}


# Age values converting
customers_df['age'] = customers_df['q1']
customers_df['age'].replace(age_group, inplace=True)

print('age Total # :', customers_df['q1'].count())
print('age Unique # :', customers_df['age'].value_counts().count())
print(customers_df['age'].value_counts(dropna=True, ascending=False))

sns.distplot(a=customers_df['age'])
plt.show()
plt.clf()


# Devices information
print('iPhone # :', customers_df.q2r1[customers_df.q2r1 == 1].count())
print('iPod # :', customers_df.q2r2[customers_df.q2r2 == 1].count())
print('Android # :', customers_df.q2r3[customers_df.q2r3 == 1].count())
print('BlackBerry # :', customers_df.q2r4[customers_df.q2r4 == 1].count())
print('Nokia # :', customers_df.q2r5[customers_df.q2r5 == 1].count())
print('Windows # :', customers_df.q2r6[customers_df.q2r6 == 1].count())
print('HP # :', customers_df.q2r7[customers_df.q2r7 == 1].count())
print('Tablet # :', customers_df.q2r8[customers_df.q2r8 == 1].count())
print('Other # :', customers_df.q2r9[customers_df.q2r9 == 1].count())
print('None # :', customers_df.q2r10[customers_df.q2r10 == 1].count())


# Creating a Multi_Device variable to see multi-device users
customers_df['Multi_Device'] = 0

customers_df['Multi_Device'] = customers_df['q2r1']\
                             + customers_df['q2r2']\
                             + customers_df['q2r3']\
                             + customers_df['q2r4']\
                             + customers_df['q2r5']\
                             + customers_df['q2r6']\
                             + customers_df['q2r7']\
                             + customers_df['q2r8']\
                             + customers_df['q2r9']\
                             + customers_df['q2r10']

print('Total Respondents # :', customers_df['Multi_Device'].count())
print('Total Device # :', customers_df['Multi_Device'].sum())
print('AVG Device # :', customers_df['Multi_Device'].mean())
print('Median Device # :', customers_df['Multi_Device'].median())
print(customers_df['Multi_Device'].value_counts(dropna=True, ascending=False))


# App usage information
print('Music # :', customers_df.q4r1[customers_df.q4r1 == 1].count())
print('TV Check-in # :', customers_df.q4r2[customers_df.q4r2 == 1].count())
print('Entertainment # :', customers_df.q4r3[customers_df.q4r3 == 1].count())
print('TV Show # :', customers_df.q4r4[customers_df.q4r4 == 1].count())
print('Game # :', customers_df.q4r5[customers_df.q4r5 == 1].count())
print('SNS # :', customers_df.q4r6[customers_df.q4r6 == 1].count())
print('News General # :', customers_df.q4r7[customers_df.q4r7 == 1].count())
print('Shopping # :', customers_df.q4r8[customers_df.q4r8 == 1].count())
print('News Specific # :', customers_df.q4r9[customers_df.q4r9 == 1].count())
print('Other # :', customers_df.q4r10[customers_df.q4r10 == 1].count())
print('None # :', customers_df.q4r11[customers_df.q4r11 == 1].count())


# Creating App_usage variable to the number of apps in general
customers_df['App_usage'] = 0

customers_df['App_usage'] = customers_df['q4r1']\
                          + customers_df['q4r2']\
                          + customers_df['q4r3']\
                          + customers_df['q4r4']\
                          + customers_df['q4r5']\
                          + customers_df['q4r6']\
                          + customers_df['q4r7']\
                          + customers_df['q4r8']\
                          + customers_df['q4r9']\
                          + customers_df['q4r10']\
                          + customers_df['q4r11']

print('Total Respondents # :', customers_df['App_usage'].count())
print('Total Apps # :', customers_df['App_usage'].sum())
print('AVG Apps # :', customers_df['App_usage'].mean())
print('Median Apps # :', customers_df['App_usage'].median())
print(customers_df['App_usage'].value_counts(dropna=True, ascending=False))


# Distribution checking with high standard deviation
std = customers_desc.loc[['std'], :]
pd.DataFrame(pd.np.transpose(std)).sort_values(ascending=False, by='std')


# Checking distributions of behavior variables with to 3 high std
sns.boxplot(customers_df['q26r11'])
plt.show()
plt.clf()


sns.boxplot(customers_df['q24r9'])
plt.show()
plt.clf()


sns.boxplot(customers_df['q24r4'])
plt.show()
plt.clf()


########################
# Correlation analysis
########################


df_corr = customers_df.corr().round(2)


# Finding strong correlations
for col in df_corr:
    list(df_corr[col])
    if ((df_corr[col].sort_values(ascending=False)[1] >= 0.60) or
       (df_corr[col].min() <= -0.60)):
        print(col, df_corr[col].sort_values(ascending=False)[1],
              df_corr[col].min())


df_corr.to_excel('correlation.xlsx')


########################
# Manipulation (Correction)
########################


# For Q.13, converted order 1-4 (Very often - Almost never) to 4-1    
new_order_q13 = {1: 4,
                 2: 3,
                 3: 2,
                 4: 1}

customers_df['q13r1'].replace(new_order_q13, inplace=True)
customers_df['q13r2'].replace(new_order_q13, inplace=True)
customers_df['q13r3'].replace(new_order_q13, inplace=True)
customers_df['q13r4'].replace(new_order_q13, inplace=True)
customers_df['q13r5'].replace(new_order_q13, inplace=True)
customers_df['q13r6'].replace(new_order_q13, inplace=True)
customers_df['q13r7'].replace(new_order_q13, inplace=True)
customers_df['q13r8'].replace(new_order_q13, inplace=True)
customers_df['q13r9'].replace(new_order_q13, inplace=True)
customers_df['q13r10'].replace(new_order_q13, inplace=True)
customers_df['q13r11'].replace(new_order_q13, inplace=True)
customers_df['q13r12'].replace(new_order_q13, inplace=True)


# For Q.24, converted order 1-6 (Agree Strongly - Disagree Strongly) to 6-1    
new_order_q242526 = {1: 6,
                     2: 5,
                     3: 4,
                     4: 3,
                     5: 2,
                     6: 1}

customers_df['q24r1'].replace(new_order_q242526, inplace=True)
customers_df['q24r2'].replace(new_order_q242526, inplace=True)
customers_df['q24r3'].replace(new_order_q242526, inplace=True)
customers_df['q24r4'].replace(new_order_q242526, inplace=True)
customers_df['q24r5'].replace(new_order_q242526, inplace=True)
customers_df['q24r6'].replace(new_order_q242526, inplace=True)
customers_df['q24r7'].replace(new_order_q242526, inplace=True)
customers_df['q24r8'].replace(new_order_q242526, inplace=True)
customers_df['q24r9'].replace(new_order_q242526, inplace=True)
customers_df['q24r10'].replace(new_order_q242526, inplace=True)
customers_df['q24r11'].replace(new_order_q242526, inplace=True)
customers_df['q24r12'].replace(new_order_q242526, inplace=True)

customers_df['q25r1'].replace(new_order_q242526, inplace=True)
customers_df['q25r2'].replace(new_order_q242526, inplace=True)
customers_df['q25r3'].replace(new_order_q242526, inplace=True)
customers_df['q25r4'].replace(new_order_q242526, inplace=True)
customers_df['q25r5'].replace(new_order_q242526, inplace=True)
customers_df['q25r6'].replace(new_order_q242526, inplace=True)
customers_df['q25r7'].replace(new_order_q242526, inplace=True)
customers_df['q25r8'].replace(new_order_q242526, inplace=True)
customers_df['q25r9'].replace(new_order_q242526, inplace=True)
customers_df['q25r10'].replace(new_order_q242526, inplace=True)
customers_df['q25r11'].replace(new_order_q242526, inplace=True)
customers_df['q25r12'].replace(new_order_q242526, inplace=True)

customers_df['q26r3'].replace(new_order_q242526, inplace=True)
customers_df['q26r4'].replace(new_order_q242526, inplace=True)
customers_df['q26r5'].replace(new_order_q242526, inplace=True)
customers_df['q26r6'].replace(new_order_q242526, inplace=True)
customers_df['q26r7'].replace(new_order_q242526, inplace=True)
customers_df['q26r8'].replace(new_order_q242526, inplace=True)
customers_df['q26r9'].replace(new_order_q242526, inplace=True)
customers_df['q26r10'].replace(new_order_q242526, inplace=True)
customers_df['q26r11'].replace(new_order_q242526, inplace=True)
customers_df['q26r12'].replace(new_order_q242526, inplace=True)
customers_df['q26r13'].replace(new_order_q242526, inplace=True)
customers_df['q26r14'].replace(new_order_q242526, inplace=True)
customers_df['q26r15'].replace(new_order_q242526, inplace=True)
customers_df['q26r16'].replace(new_order_q242526, inplace=True)
customers_df['q26r17'].replace(new_order_q242526, inplace=True)
customers_df['q26r18'].replace(new_order_q242526, inplace=True)


"""
Kisuc Kim:
    * 88 variables of type int64 in original dataset
    * no missing values
    * no negative values
    * one demographic distribution seem skewed extremely (Race)
    * male, female are almost equally distributed 
    * 781 iphone users (50%), 374 iPod (24%), Some +a% iPad in Tablet 301 (19%)
    * 565 Android users (36%), Some +a% android tablet in Tablet 301 (19%)
    * 60% Single device user, 40% Multi devices user
    * 4.64 apps in average (4 apps in median)
    * there are some interesting correlations.
      1. 0.62 Correlation btw IPod users & Multi devices users
      2. 0.60 Correlation btw Tablet users & Multi devices users
      3. 0.71 Correlation btw luxury brands fan & designer brands fan
"""


###############################################################################
# Model Code: Agglomerative Clustering
###############################################################################


########################
# Scaling with behavior variables
########################


# Removing Case_ID, and demographic variables
customer_features_reduced = customers_df.iloc[:, 23:-14]


# Removing unnecessary behavior variables
customer_features_reduced = customer_features_reduced.drop(['q11', 'q12'], axis=1)


# Concatenating the number of devices, apps variables
customer_features_reduced = pd.concat(
    [customers_df.loc[:, ['Multi_Device', 'App_usage']],
     customer_features_reduced], axis=1)


# Scale to get equal variance
print(pd.np.var(customer_features_reduced))


scaler = StandardScaler()


scaler.fit(customer_features_reduced)


X_scaled_reduced = scaler.transform(customer_features_reduced)


X_scaled_reduced_df = pd.DataFrame(X_scaled_reduced)


print(pd.np.var(X_scaled_reduced_df))


########################
# hierarchical clustering
########################

# Building a Dendrogram
standard_mergings_ward = linkage(y=X_scaled_reduced_df,
                                 method='ward')

fig, ax = plt.subplots(figsize=(12, 12))

dendrogram(Z=standard_mergings_ward,
           leaf_rotation=90,
           leaf_font_size=7)

plt.savefig('standard_hierarchical_clust_ward.png')

plt.show()


"""
Kisuc Kim:
    * Segmented automatically
    * Hard to find actionable insights
"""


###############################################################################
# Model Code: Principal Component Analisys
###############################################################################


########################
# Run PCA without limiting the number of components
########################

customer_pca_reduced = PCA(n_components=None,
                           random_state=508)


customer_pca_reduced.fit(X_scaled_reduced)


X_pca_reduced = customer_pca_reduced.transform(X_scaled_reduced)


########################
# Analyze the scree plot to determine how many components to retain
########################

fig, ax = plt.subplots(figsize=(10, 8))


features = range(customer_pca_reduced.n_components_)


plt.plot(features,
         customer_pca_reduced.explained_variance_ratio_,
         linewidth=2,
         marker='o',
         markersize=10,
         markeredgecolor='black',
         markerfacecolor='grey')


plt.title('Research Servey PCA Scree Plot')
plt.xlabel('PCA feature')
plt.ylabel('Explained Variance')
plt.xticks(features)
plt.show()


# Finding the optimal number of components from above scree plot
customer_pca_reduced.explained_variance_ratio_.round(2)

N_components = 4

sum_components = list(range(N_components))

print(customer_pca_reduced.explained_variance_ratio_[sum_components].sum())


"""
Kisuc Kim:
    * Over 40% of data can be covered using 4 principal components
"""

########################
# Run PCA again based on the desired number of components
########################


customer_pca_reduced = PCA(n_components=4,
                           random_state=508)


customer_pca_reduced.fit(X_scaled_reduced)


########################
# Analyze factor loadings to understand principal components
########################


factor_loadings_df = pd.DataFrame(
    pd.np.transpose(
        customer_pca_reduced.components_))


factor_loadings_df = factor_loadings_df.set_index(
    customer_features_reduced.columns)


factor_loadings_df.to_excel('practice_factor_loadings.xlsx')


########################
# Analyze factor strengths per customer
########################


X_pca_reduced = customer_pca_reduced.transform(X_scaled_reduced)


X_pca_df = pd.DataFrame(X_pca_reduced)


########################
# Rename your principal components and reattach demographic information
########################

# Based on results in factor_loadings_df, Renamed each component 
X_pca_df.columns = ['Conservative',
                    'Follower',
                    'YOLO',
                    'Smart Consumer']


print(X_pca_df.head(n=5))


###############################################################################
# Model Code: KMeans Clustering after PCA
###############################################################################


########################
# One more time, scale to get equal variance
########################


print(pd.np.var(X_pca_df))


scaler = StandardScaler()


scaler.fit(X_pca_df)


X_pca_clust = scaler.transform(X_pca_df)


X_pca_clust_df = pd.DataFrame(X_pca_clust)


X_pca_clust_df.columns = X_pca_df.columns


print(pd.np.var(X_pca_clust_df))


########################
# Plotting Intertia
########################


# How many clusters do I need? Checking using the metric inertia
ks = range(1, 50)
inertias = []


for k in ks:
    # Create a KMeans instance with k clusters: model
    model = KMeans(n_clusters=k)

    # Fit model to samples
    model.fit(X_pca_clust_df)

    # Append the inertia to the list of inertias
    inertias.append(model.inertia_)


# Plot ks vs inertias
fig, ax = plt.subplots(figsize=(12, 8))
plt.plot(ks, inertias, '-o')


plt.xlabel('number of clusters, k')
plt.ylabel('inertia')
plt.xticks(ks)

plt.show()


########################
# Experiment with the desired number of components
########################


customers_k_pca = KMeans(n_clusters=5,
                         random_state=508)


customers_k_pca.fit(X_pca_clust_df)


print(customers_k_pca.inertia_)


customers_kmeans_pca = pd.DataFrame({'cluster': customers_k_pca.labels_})


print(customers_kmeans_pca.iloc[:, 0].value_counts())


########################
# Analyze cluster centers
########################


centroids_pca = customers_k_pca.cluster_centers_


centroids_pca_df = pd.DataFrame(centroids_pca)


# Renaming columns
centroids_pca_df.columns = X_pca_df.columns


print(centroids_pca_df)


# Sending data to Excel
centroids_pca_df.to_excel('customers_pca_centriods.xlsx')


########################
# Analyze cluster memberships
########################


clst_pca_df = pd.concat([customers_kmeans_pca,
                         X_pca_clust_df],
                        axis=1)

# Rename clusters
N_clusters = {0: 1,
              1: 2,
              2: 3,
              3: 4,
              4: 5}


clst_pca_df['cluster'].replace(N_clusters, inplace=True)


print(clst_pca_df)


########################
# Reattach demographic information
########################


# Attaching age, gender features
final_pca_clust_df = pd.concat(
    [customers_df.loc[:, ['age', 'q57']], clst_pca_df], axis=1)


print(final_pca_clust_df)


final_pca_clust_df.to_excel('final_pca_clust_df.xlsx')


###############################################################################
# Code for Data Analysis: Analyze in more detail
###############################################################################


# Adding a productivity step
final_pca_clust_df2050 = final_pca_clust_df[(
    final_pca_clust_df.age < 60) & (final_pca_clust_df.age > 10)]


data_df = final_pca_clust_df2050


# Renaming gender
Gender = {1: 'Male',
          2: 'Female'}


data_df['Gender'] = data_df['q57']
data_df['Gender'].replace(Gender, inplace=True)


########################
# Age
########################


# Conservative
fig, ax = plt.subplots(figsize=(8, 4))
sns.boxplot(x='age',
            y='Conservative',
            hue='cluster',
            data=data_df)

plt.ylim(-5, 5)
plt.tight_layout()
plt.show()


# Follower
fig, ax = plt.subplots(figsize=(8, 4))
sns.boxplot(x='age',
            y='Follower',
            hue='cluster',
            data=data_df)

plt.ylim(-5, 5)
plt.tight_layout()
plt.show()


# YOLO
fig, ax = plt.subplots(figsize=(8, 4))
sns.boxplot(x='age',
            y='YOLO',
            hue='cluster',
            data=data_df)

plt.ylim(-5, 5)
plt.tight_layout()
plt.show()


# Smart Consumer
fig, ax = plt.subplots(figsize=(8, 4))
sns.boxplot(x='age',
            y='Smart Consumer',
            hue='cluster',
            data=data_df)

plt.ylim(-5, 5)
plt.tight_layout()
plt.show()


########################
# Gender
########################


# Conservative
fig, ax = plt.subplots(figsize=(8, 4))
sns.boxplot(x='Gender',
            y='Conservative',
            hue='cluster',
            data=data_df)

plt.ylim(-5, 5)
plt.tight_layout()
plt.show()


# Follower
fig, ax = plt.subplots(figsize=(8, 4))
sns.boxplot(x='Gender',
            y='Follower',
            hue='cluster',
            data=data_df)

plt.ylim(-5, 5)
plt.tight_layout()
plt.show()


# YOLO
fig, ax = plt.subplots(figsize=(8, 4))
sns.boxplot(x='Gender',
            y='YOLO',
            hue='cluster',
            data=data_df)

plt.ylim(-5, 5)
plt.tight_layout()
plt.show()


# Smart Consumer
fig, ax = plt.subplots(figsize=(8, 4))
sns.boxplot(x='Gender',
            y='Smart Consumer',
            hue='cluster',
            data=data_df)

plt.ylim(-5, 5)
plt.tight_layout()
plt.show()


########################
# Purpose
########################
# So, What app we are going to develop for? 


# Creating new data frame with cluster variable
Purpose_df = final_pca_clust_df.loc[:, ['cluster']]


# Creating new variables depending on purpose
Purpose_df['Entertainmemt'] = 0

Purpose_df['Entertainmemt'] = customers_df['q4r3']\
                            + customers_df['q4r5']\
                            + customers_df['q13r5']\
                            + customers_df['q13r9']\
                            + customers_df['q13r10']
                            
                            
Purpose_df['SNS'] = 0

Purpose_df['SNS'] = customers_df['q4r6']\
                  + customers_df['q13r1']\
                  + customers_df['q13r2']\
                 + customers_df['q13r11']


Purpose_df['News'] = 0

Purpose_df['News'] = customers_df['q4r7']\
                   + customers_df['q4r9']
                
                
Purpose_df['Shopping'] = 0

Purpose_df['Shopping'] = customers_df['q4r8']


Purpose_df['Music'] = 0

Purpose_df['Music'] = customers_df['q4r1']\
                    + customers_df['q13r3']\
                    + customers_df['q13r4']\
                    + customers_df['q13r7']\
                    + customers_df['q13r8']


Purpose_df['TV'] = 0

Purpose_df['TV'] = customers_df['q4r2']\
                 + customers_df['q4r4']\
                 + customers_df['q13r6']\
                 + customers_df['q13r12']


# One-Hot encoding for cluster
cluster_dummies = pd.get_dummies(list(final_pca_clust_df['cluster']), drop_first = False)


# Concatenating One-Hot Encoded Values with the Larger DataFrame
Purpose_df = pd.concat([Purpose_df.loc[:,:],cluster_dummies],axis = 1)


Purpose_df.to_excel('Purpose_df.xlsx')


# Drop cluster variable
Final_Purpose_df = Purpose_df.drop(['cluster'], axis=1)


##############################################################################
# Code for Data Analysis: Correlation Analysis
##############################################################################


# Using correlation to identify cluster's main purpose of using app
Final_df_corr = Final_Purpose_df.corr().round(2)


print(Final_df_corr)


Final_df_corr.to_excel('Final_df_corr.xlsx')


########################
# Correlation Heatmap
########################

# Using palplot to view a color scheme
sns.palplot(sns.color_palette('coolwarm', 12))

fig, ax = plt.subplots(figsize=(12, 12))

sns.heatmap(Final_df_corr,
            cmap='coolwarm',
            annot=True,
            linewidths=0.1)

plt.show()


"""
Kisuc Kim:
    * there is positive correlation btw Entertainment & cluster 5.
    * there is positive correlation btw Music & cluster 5.
"""



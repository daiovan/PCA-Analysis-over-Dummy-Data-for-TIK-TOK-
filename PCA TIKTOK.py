#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


# In[2]:


df = pd.read_csv('tiktok_influencer_dataset.csv')
df.head()  # Check first 5 rows


# In[3]:


df.describe()


# In[4]:


#Step 1: Standerdize the data


# In[5]:


scaler = StandardScaler()
X_scaled = scaler.fit_transform(df)


# In[6]:


df_scaled = pd.DataFrame(X_scaled, columns=df.columns)
df_scaled.head()


# In[7]:


# Assuming you already have df_scaled from Step 1:
cov_matrix = np.cov(df_scaled.T)  # .T transposes the DataFrame to get features as rows


# In[8]:


cov_df = pd.DataFrame(cov_matrix, columns=df.columns, index=df.columns)
cov_df


# In[9]:


import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(6, 5))
sns.heatmap(cov_df, annot=True, cmap='coolwarm', fmt=".2f", square=True)
plt.title("Covariance Matrix Heatmap")
plt.show()


# In[10]:


# Step 3: Eigen Decomposition of Covariance Matrix
eig_vals, eig_vecs = np.linalg.eig(cov_matrix)


# In[11]:


print("Eigenvalues:")
print(eig_vals)


# In[12]:


print("Eigenvectors (columns):")
print(eig_vecs)


# In[13]:


pcs_df = pd.DataFrame({
    'Principal Component': [f'PC{i+1}' for i in range(len(eig_vals))],
    'Eigenvalue': eig_vals,
    'Explained Variance Ratio': eig_vals / eig_vals.sum(),
    'Eigenvector': [eig_vecs[:, i] for i in range(len(eig_vals))]
})

pcs_df


# In[14]:


#step 4
eig_vals = [2.727078, 0.20459684, 0.07433718]
explained_variance_ratio = eig_vals / np.sum(eig_vals)
print("Explained Variance Ratio:", explained_variance_ratio)


# In[15]:


# Keep top 2 PCs (eigenvectors are columns)
feature_vector = eig_vecs[:, :2]  # shape: (3 features × 2 components)
print("Feature Vector (matrix of top 2 eigenvectors):\n", feature_vector)


# In[16]:


#step 5 
#Assuming you already have X_scaled (standardized data) and feature_vector (top 2 eigenvectors)
final_data = np.dot(X_scaled, feature_vector)

# Optional: turn into DataFrame for plotting or analysis
final_df = pd.DataFrame(final_data, columns=['PC1', 'PC2'])
final_df.head()


# In[17]:


import matplotlib.pyplot as plt

plt.figure(figsize=(8, 6))
plt.scatter(final_df['PC1'], final_df['PC2'], alpha=0.7, color='orange', edgecolor='k')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('TikTok Influencer PCA Projection')
plt.grid(True)
plt.show()


# """
# ============================================
# INTERPRETING PCA OUTPUT — EIGENVALUE DECOMPOSITION
# ============================================
# 
# After fitting the PCA model, we extracted two important outputs:
# 
# 1. pca.explained_variance_
# 2. pca.components_
# 
# --------------------------------------------
# 1. pca.explained_variance_
# --------------------------------------------
# This gives us the **eigenvalues** of the covariance matrix of our standardized data.
# 
# Each eigenvalue corresponds to one **principal component** (PC) and tells us **how much variance** is captured along that new axis.
# 
# For example, if you see:
#     [2.73, 0.20]
# It means:
# - The first principal component (PC1) explains 2.73 units of variance
# - The second principal component (PC2) explains 0.20 units
# 
# To find the **percentage of variance** explained by each component:
#     pca.explained_variance_ratio_
# 
# --------------------------------------------
# 2. pca.components_
# --------------------------------------------
# This returns the **eigenvectors** of the covariance matrix — each one is a **principal axis** in the new feature space.
# 
# Each row corresponds to a principal component.
# Each column shows how much that original feature contributes to the component.
# 
# Example:
#     [[ 0.58, 0.57, 0.58],   ← PC1
#      [-0.52, 0.80, -0.27]]  ← PC2
# 
# Here:
# - PC1 is a blend of all 3 features (Views, Likes, Comments), almost equally weighted
# - PC2 contrasts Likes positively with Views and Comments negatively
# 
# These eigenvectors are the directions where the data varies the most — the **principal directions** of the data cloud.
# 
# --------------------------------------------
# Relating to Theory:  A·v = λ·v
# --------------------------------------------
# In PCA, the matrix A is the **covariance matrix** of the standardized data.
# 
# - The PCA algorithm performs **eigen decomposition** of this matrix.
# - Each row in `pca.components_` is a vector v (an eigenvector)
# - Each value in `pca.explained_variance_` is a λ (eigenvalue)
# 
# So:
#     (Covariance Matrix) · (Eigenvector) = (Eigenvalue) · (Eigenvector)
# 
# This gives us the **axes of maximum variance**, sorted by importance.
# These directions become the new feature space where we project the data.
# 
# --------------------------------------------
# Why It Matters
# --------------------------------------------
# By selecting just the top 1 or 2 components:
# - We reduce dimensionality
# - We retain most of the data’s structure
# - We make it easier to visualize, analyze, and cluster influencers
# 
# In our case, PC1 + PC2 captured over **97.5%** of the total variance — so we retained nearly all the signal while dropping noise.
# """

# In[ ]:





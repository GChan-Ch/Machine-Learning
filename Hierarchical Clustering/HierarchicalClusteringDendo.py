from sklearn import datasets 
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
from sklearn.preprocessing import StandardScaler 
from scipy.cluster.hierarchy import cophenet, dendrogram, linkage 
from scipy.spatial.distance import pdist 
 
iris = datasets.load_iris() 
 
# Let's convert to dataframe 
iris = pd.DataFrame(data=np.c_[iris['data'], iris['target']], 
                    columns=iris['feature_names'] + ['species']) 
 
# Let's remove spaces from column name 
iris.columns = iris.columns.str.replace(' ', '') 
iris.head()
 
# independent variables, dimana data mulai dari kolom 0 s.d 3 
X = iris.iloc[:, :3] 
y = iris.species
sc = StandardScaler() 
sc.fit(X) 
X = sc.transform(X) 
 
# generate the linkage matrix 
Z = linkage(X, 'ward') 
c, coph_dists = cophenet(Z, pdist(X)) 
 
# calculate full dendrogram 
plt.figure(figsize=(25, 10)) 
plt.title('Agglomerative Hierarchical Clustering Dendrogram') 
plt.xlabel('sample index') 
plt.ylabel('distance') 
dendrogram( 
    Z, 
    leaf_rotation=90.,  # rotates the x axis labels 
    leaf_font_size=8.,  # font size for the x axis labels 
) 
plt.tight_layout() 
plt.show() 
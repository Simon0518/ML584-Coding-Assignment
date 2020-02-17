import pandas as pd
import numpy as np
from kmodes.kmodes import KModes

cars_data = pd.read_csv(r'cars.csv')

# counts for frequency
type_counts = cars_data['Type'].value_counts(dropna=False)
origin_counts = cars_data['Origin'].value_counts(dropna=False)
driveTrain_counts = cars_data['DriveTrain'].value_counts(dropna=False)
cylinder_counts = cars_data['Cylinders'].value_counts(dropna=False)

# Question 2a
print("The frequency of feature type is showed below:")
print(type_counts)
print()

# Question 2b
print("The frequency of feature drivetrain is showed below:")
print(driveTrain_counts)
print()

# Question 2c
asia_eu_distance = 1 / origin_counts['Asia'] + 1 / origin_counts['Europe']
print(r"The distance metric between 'Asia' and 'Europe' is: %.4f" % asia_eu_distance)
print()

# Question 2d
five_cylinders_nan_distance = 1 / cylinder_counts[5.0] + 1 / cylinder_counts[np.nan]
print("The distance between 5 cylinders and nan is: %.4f" % five_cylinders_nan_distance)
print()

# Question 2e
# convert columns to categorical data type
cars_data['Type'] = cars_data['Type'].astype('category')
cars_data['Origin'] = cars_data['Origin'].astype('category')
cars_data['DriveTrain'] = cars_data['DriveTrain'].astype('category')
cars_data['Cylinders'] = cars_data['Cylinders'].astype('category')

# encode categorical data for columns
# NaN value will be encoded to -1
cat_col = cars_data.select_dtypes(['category']).columns
df = cars_data[cat_col].apply(lambda x: x.cat.codes)

# fit model
km = KModes(n_clusters=3, init='Huang', random_state=555)
clusters = km.fit(df)

# get results
cents = km.cluster_centroids_
predict_results = km.predict(df)
unique, counts = np.unique(predict_results, return_counts=True)
num_obs_in_each_cluster = dict(zip(unique, counts))
print("Number of observations in cluster 1: %d" % num_obs_in_each_cluster[0])
print("Number of observations in cluster 2: %d" % num_obs_in_each_cluster[1])
print("Number of observations in cluster 3: %d" % num_obs_in_each_cluster[2])

cluster_num = 1
for i in cents:
    index = 0
    cent_in_text = []
    for s in cat_col:
        d = dict(enumerate(cars_data[s].cat.categories))
        cent_in_text.append(d[i[index]])
        index += 1
    print("Cluster %d: " % cluster_num, cent_in_text)
    cluster_num += 1
print()

# Question 2f
df = cars_data[cat_col].copy()
df['Cluster'] = predict_results
cluster1_df = df.loc[df['Cluster'] == 0]
cluster2_df = df.loc[df['Cluster'] == 1]
cluster3_df = df.loc[df['Cluster'] == 2]
print("Cluster 1 frequency distribution of feature origin:")
print(cluster1_df['Origin'].value_counts())
print()
print("Cluster 2 frequency distribution of feature origin:")
print(cluster2_df['Origin'].value_counts())
print()
print("Cluster 3 frequency distribution of feature origin:")
print(cluster3_df['Origin'].value_counts())
print()

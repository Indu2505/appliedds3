import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.stats import t
from sklearn.preprocessing import scale
import scipy.optimize as opt
import errors as err

# The below code is used for convert the dataset into h years as columns and one with countries as columns
def ConvertDataset(filepath):
    '''
    The ConvertDataset function uses input parameter filepath to return datasets wuth years as columns and one with countries as columns
    ----------
    filepath : String
        the file location path is given as the input parameter.
    Returns
    -------
    yearDataset : dataframe
        years as the column names dataset
        
    countryDataset : dataframe
        countries as the column names dataset
    '''
    
    #reading the dataset.
    originalDataset = pd.read_csv(filepath)
    
    #dropping the columns with are not required and creating the year dataset.
    yearDataset = originalDataset.drop(columns=['Indicator Code','Country Code'])
    
    #Creating countryDataset
    countryDataset = pd.DataFrame.transpose(yearDataset)
    columnNames = countryDataset.iloc[0].values.tolist()
    countryDataset.columns = columnNames
    countryDataset = countryDataset.iloc[1:]
    return yearDataset, countryDataset

yearDataset , countryDataset = ConvertDataset('climate_change.csv')

indicators = countryDataset.iloc[0].unique()

ukdata = countryDataset[['United Kingdom']]

co2emission = ukdata.iloc[31:62,44]
forestarea = ukdata.iloc[31:62,67]

uk_new_df = pd.DataFrame({
    'CO2 Emission': co2emission.values,
    'Forest Area': forestarea.values
},index=co2emission.index)


# Apply K-Means clustering
kmeans = KMeans(n_clusters=3)  # You can choose the number of clusters based on your dataset
uk_new_df['Cluster'] = kmeans.fit_predict(uk_new_df)

# Plotting the clusters
plt.scatter(uk_new_df['CO2 Emission'], uk_new_df['Forest Area'], c=uk_new_df['Cluster'], cmap='viridis', edgecolor='k', s=50)
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], c='red', marker='X', s=200, label='Cluster Centers')
plt.title('K-Means Clustering')
plt.xlabel('CO2 Emission')
plt.ylabel('Forest Area')
plt.legend()
plt.show()


def logistics(t, a, k, t0):
    """ Computes logistics function with scale and incr as free parameters
    """
    f = a / (1.0 + np.exp(-k * (t - t0)))
    return f

plt.figure()
plt.plot(uk_new_df.index.values, uk_new_df['CO2 Emission'], label="data")
plt.legend()
plt.title("CO2 emission in UK")
plt.xlabel('Year')
plt.ylabel('CO2 emissions')
plt.xticks(rotation=90)
plt.show()


popt, pcorr = opt.curve_fit(logistics, uk_new_df.index.values.astype(float), uk_new_df['CO2 Emission'],p0=(16e8, 0.04, 1985.0))
print("Fit parameter", popt)

uk_new_df["pop_logistics"] = logistics(uk_new_df.index.values.astype(float), *popt)
plt.figure()
plt.title("logistics function")
plt.plot(uk_new_df.index.values, uk_new_df['CO2 Emission'], label="data")
plt.plot(uk_new_df.index.values, uk_new_df["pop_logistics"], label="fit")
plt.legend()
plt.xticks(rotation=90)
plt.xlabel('Year')
plt.ylabel('CO2 emissions')
plt.show()



plt.figure()
plt.plot(uk_new_df.index.values, uk_new_df['Forest Area'], label="data")
plt.legend()
plt.title("'Forest Area' in UK")
plt.xlabel('Year')
plt.ylabel('Forest area')
plt.xticks(rotation=90)
plt.show()


popt, pcovar = opt.curve_fit(logistics, uk_new_df.index.values.astype(float), uk_new_df['Forest Area'],p0=(16e8, 0.04, 1985.0))
print("Fit parameter", popt)

uk_new_df["pop_logistics"] = logistics(uk_new_df.index.values.astype(float), *popt)
plt.figure()
plt.title("logistics function")
plt.plot(uk_new_df.index.values, uk_new_df['Forest Area'], label="data")
plt.plot(uk_new_df.index.values, uk_new_df["pop_logistics"], label="fit")
plt.legend()
plt.xticks(rotation=90)
plt.xlabel('Year')
plt.ylabel('Forest Area')
plt.show()



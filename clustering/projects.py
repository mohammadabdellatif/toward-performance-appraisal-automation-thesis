import numpy as np
from pandas import DataFrame
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


class ProjectsClustering:

    def cluster_projects(self, projects_sumary_df: DataFrame, n: int = 5):
        df = projects_sumary_df.copy()
        self.__scale_records(df)
        labels = self.__algorithm(df, n)
        return labels

    def __scale_records(self, aggregated):
        for key in aggregated.select_dtypes(include=np.number):
            scaler = StandardScaler()
            aggregated[key] = scaler.fit_transform(aggregated[key].values.reshape(-1, 1)).round(decimals=3)

    def __algorithm(self, aggregated: DataFrame, n: int):
        kmeans: KMeans = KMeans(n_clusters=n, random_state=42, n_init="auto").fit(
            aggregated.drop(columns='issue_proj'))
        return kmeans.labels_

import numpy as np
from pandas import DataFrame
from sklearn.cluster import AgglomerativeClustering, KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler


class IssuesClustering:

    def cluster(self, n: int, issues_df: DataFrame, features: list[str]):
        return self.__cluster(n, issues_df, features, self.__kmeans)

    def hierarchical(self, n: int, issues_df: DataFrame, features: list[str]):
        return self.__cluster(n, issues_df, features, self.__agglomerative)

    def __agglomerative(self, df: DataFrame, n: int, issues_df, features):
        clustering = AgglomerativeClustering(n_clusters=n, linkage='ward')
        clustering.fit(df)
        return clustering.labels_

    def __cluster(self, n: int, issues_df: DataFrame, features: list[str], algo: object):
        df = issues_df[features].copy()
        self.__standardize_features(df)
        return algo(df, n, issues_df, features)

    def __kmeans(self, df, n, issues_df, features):
        km = KMeans(n_clusters=n, n_init='auto', random_state=42)
        km.fit(df)
        # TODO this should be returned somehow not printed
        for centers in km.cluster_centers_:
            center_vals = []
            for i, center in enumerate(centers):
                f = features[i]
                f_mean = issues_df[f].mean()
                f_std = issues_df[f].std()
                center_val = round(center * f_std + f_mean, 2)
                center_vals.append(center_val)
            print(center_vals)
        return km.labels_

    def __dbscan(self, df, n, issues_df, features):
        db = DBSCAN(eps=0.5, min_samples=100).fit(df)
        return db.labels_

    def __standardize_features(self, df):
        for c in df.select_dtypes(include=np.number):
            df.loc[:, c] = StandardScaler().fit_transform(df[c].values.reshape(-1, 1))

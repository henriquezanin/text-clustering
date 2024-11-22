import string
import nltk
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('rslp')
from nltk.tokenize import word_tokenize
from nltk.stem.porter import *
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

class DataFrame:

    def __init__(self, dataset: pd.DataFrame, target_column:str):
        self._dataset = dataset.copy(deep=True)
        self.values = pd.DataFrame()
        self.values['target'] = self._dataset[target_column]
        self.target = self.values['target']
        self.embeddings = None
        self.clusters = None
        self.model = None
        self.silhouette_sample = None
        self.silhouette_score = None
        self.silhouette_sample_mean = None
        self.silhouette_score_mean = None
        self.cluster_centers =  None

    def set_index(self, index:str):
        self.values[index] = self._dataset[index]
        self.values.set_index(index, inplace=True)

    def set_embeddings(self, embeddings:list):
        self.values['embeddings'] = embeddings
        self.embeddings = self.values['embeddings']

    def set_cluster(self, cluster:list):
        self.values['cluster'] = cluster
        self.clusters = self.values['cluster']
    
    def Kmeans(self,n_clusters=5, random_state=0, n_init=10):
        self.model = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=n_init)
        self.model.fit(self.values.embeddings.to_list())
        self.set_cluster(self.model.labels_)
        self.silhouette_sample = silhouette_samples(self.values.embeddings.to_list(), self.model.labels_)
        self.silhouette_sample_mean = self.silhouette_sample.mean()
        self.silhouette_score = silhouette_score(self.values.embeddings.to_list(), self.model.labels_)
        self.silhouette_score_mean = self.silhouette_score.mean()
        self.cluster_centers = self.model.cluster_centers_

    def DBScan(self, eps=0.05, min_sample=5, metric='cosine'):
        self.model = DBSCAN(eps=eps, min_samples=min_sample, metric=metric)
        self.model.fit(self.values.embeddings.to_list())
        self.set_cluster(self.model.labels_)
        self.silhouette_sample = silhouette_samples(self.values.embeddings.to_list(), self.model.labels_).mean()
        self.silhouette_sample_mean = self.silhouette_sample.mean()
        self.silhouette_score = silhouette_score(self.values.embeddings.to_list(), self.model.labels_).mean()
        self.silhouette_score_mean = self.silhouette_score.mean()

    def ClustersDistances(self):
        distances = list()
        for outer in range(len(self.model.cluster_centers_)):
            inner_distances = list()
            for inner in range(len(self.model.cluster_centers_)):
                dist = np.linalg.norm(self.model.cluster_centers_[outer] - self.model.cluster_centers_[inner])
                inner_distances.append(dist)
            distances.append(inner_distances)
        return distances

    def BestK(self, max_clusters=10, plot=True):
        values = list()
        best_k = 0
        best_silhouette = 0
        for k in tqdm(range(2,max_clusters+1)):
            self.Kmeans(n_clusters=k)
            values.append([k, self.silhouette_sample_mean])
            if self.silhouette_sample_mean > best_silhouette:
                best_silhouette = self.silhouette_sample_mean
                best_k = k
        df = pd.DataFrame(values).set_index(0)
        if plot == True:
            df.plot(title="Silhouette Samples", xlabel="Cluster size (k)", ylabel="Silhouette Value")
            plt.show()
        return (best_k, best_silhouette), df
    
    def BestEPS(self, initial_eps=0.05,max_eps=1,increment=0.05, plot=True):
        values = list()
        best_eps = 0
        best_silhouette = 0
        eps = initial_eps
        while eps <= max_eps:
            self.DBScan(eps=eps)
            values.append([eps, self.silhouette_sample_mean])
            if self.silhouette_sample > best_silhouette:
                best_silhouette = self.silhouette_sample_mean
                best_eps = eps
            print(eps, self.silhouette_sample)
            eps = eps + increment
        df = pd.DataFrame(values).set_index(0)
        if plot == True:
            df.plot(title="Silhouette Samples", xlabel="EPS value", ylabel="Silhouette Value")
            plt.show()
        return (best_eps, best_silhouette), df
    
    def SilhouettePlot(self, range_n_clusters:list):
        for n_clusters in tqdm(range_n_clusters):
            # Create a subplot with 1 row and 2 columns
            fig, ax1 = plt.subplots(1, 1)
            fig.set_size_inches(18, 18)
            # The 1st subplot is the silhouette plot
            # The silhouette coefficient can range from -1, 1 but in this example all
            # lie within [-0.1, 1]
            ax1.set_xlim([-0.1, 1])
            # The (n_clusters+1)*10 is for inserting blank space between silhouette
            # plots of individual clusters, to demarcate them clearly.
            ax1.set_ylim([0, len(self.embeddings) + (n_clusters + 1) * 10])
            # Initialize the clusterer with n_clusters value and a random generator
            # seed of 10 for reproducibility.
            self.Kmeans(n_clusters=n_clusters)
            # The silhouette_score gives the average value for all the samples.
            # This gives a perspective into the density and separation of the formed
            # clusters
            silhouette_avg = self.silhouette_score_mean
            print(
                "For n_clusters =",
                n_clusters,
                "The average silhouette_score is :",
                silhouette_avg,
            )
            # Compute the silhouette scores for each sample
            sample_silhouette_values = self.silhouette_sample
            y_lower = 10
            for i in range(n_clusters):
                # Aggregate the silhouette scores for samples belonging to
                # cluster i, and sort them
                ith_cluster_silhouette_values = sample_silhouette_values[self.clusters == i]
                ith_cluster_silhouette_values.sort()
                size_cluster_i = ith_cluster_silhouette_values.shape[0]
                y_upper = y_lower + size_cluster_i
                color = cm.nipy_spectral(float(i) / n_clusters)
                ax1.fill_betweenx(
                    np.arange(y_lower, y_upper),
                    0,
                    ith_cluster_silhouette_values,
                    facecolor=color,
                    edgecolor=color,
                    alpha=0.7,
                )
                # Label the silhouette plots with their cluster numbers at the middle
                ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
                # Compute the new y_lower for next plot
                y_lower = y_upper + 10  # 10 for the 0 samples
            ax1.set_title("The silhouette plot for the various clusters.")
            ax1.set_xlabel("The silhouette coefficient values")
            ax1.set_ylabel("Cluster label")
            # The vertical line for average silhouette score of all the values
            ax1.axvline(x=silhouette_avg, color="red", linestyle="--")
            ax1.set_yticks([])  # Clear the yaxis labels / ticks
            ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])
            plt.suptitle(
                "Silhouette analysis for KMeans clustering on sample data with n_clusters = %d"
                % n_clusters,
                fontsize=14,
                fontweight="bold",
            )
        plt.show()


class Preprocessing:

    def __init__(self, dataset: pd.DataFrame, target_column: str, index_column: str):
        self._dataframe = DataFrame(dataset, target_column)
        if index_column != "":
            self._dataframe.set_index(index_column)
    
    def __removeStopWords(self, text,stop_words):
        s = str(text).lower() # tudo para caixa baixa
        table = str.maketrans({key: None for key in string.punctuation})
        s = s.translate(table) # remove pontuacao
        tokens = word_tokenize(s) #obtem tokens
        v = [i for i in tokens if not i in stop_words and not i.isdigit()] # remove stopwords
        s = ""
        for token in v:
            s += token+" "
        return s.strip()

    def __stemming(self,tokens,stemmer):
        tokens = word_tokenize(tokens, language='portuguese')
        sentence_stem = ''
        doc_text_stems = [stemmer.stem(i) for i in tokens]
        for stem in doc_text_stems:
            sentence_stem += stem+" "
        return sentence_stem.strip()


    def __createTokens(self, dataset, stop_words=nltk.corpus.stopwords.words('portuguese'), stemmer=nltk.stem.RSLPStemmer()):
        d = []
        for row in dataset:
            text2 = self.__removeStopWords(row,stop_words)
            text3 = self.__stemming(text2, stemmer)
            d.append(text3)
        return d
    
    def __tfidf(self, min_df,ngram_range):
        vsm = TfidfVectorizer(tokenizer=self.__createTokens,min_df=min_df,ngram_range=ngram_range)
        X = vsm.fit_transform(self._dataframe.target)
        vsm = pd.DataFrame(X.todense(), columns=vsm.get_feature_names_out())
        X = np.array(vsm)
        length = np.sqrt((X**2).sum(axis=1))[:,None]
        X = X / length
        return X

    def Tfidf(self, min_df:int=3, ngram_range:list=(2,2)):
        X = self.__tfidf(min_df, ngram_range)
        self._dataframe.set_embeddings(X.tolist())
        return self._dataframe

    def Bert(self,model="sentence-transformers/distiluse-base-multilingual-cased-v2"):
        model = SentenceTransformer(model)
        embeddings = list(model.encode(self._dataframe.target.to_list()))
        self._dataframe.set_embeddings(embeddings)
        return self._dataframe
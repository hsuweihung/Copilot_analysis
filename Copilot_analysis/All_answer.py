import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import umap.umap_ as umap
import hdbscan
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.metrics import silhouette_score

df = pd.read_excel("C:\\Users\\William_Hsu\\Desktop\\PDR_Copilot\\User_all_qusetion.xlsx")

essential_prompts = df[df['Versiontype'] == 'Essential']['Prompt'].dropna().values
subscription_prompts = df[df['Versiontype'] == 'Subscription']['Prompt'].dropna().values
print(f"Essential prompts count: {len(essential_prompts)}")
print(f"Subscription prompts count: {len(subscription_prompts)}")

vectorizer = CountVectorizer(max_df=0.9, min_df=5, stop_words='english')

all_prompts = np.concatenate([essential_prompts, subscription_prompts])

X_all = vectorizer.fit_transform(all_prompts)

X_essential = X_all[:len(essential_prompts)]
X_subscription = X_all[len(essential_prompts):]

reducer_essential = umap.UMAP(n_neighbors=15, n_components=10, random_state=42)
X_umap_essential = reducer_essential.fit_transform(X_essential.toarray())

clusterer_essential = hdbscan.HDBSCAN(min_cluster_size=20, min_samples=5)
labels_essential = clusterer_essential.fit_predict(X_umap_essential)
optimal_k_essential = len(set(labels_essential)) - (1 if -1 in labels_essential else 0)
optimal_k_essential = min(optimal_k_essential, 10)
print(f"Essential最佳主題數量（n_components）: {optimal_k_essential}")

lda_essential = LatentDirichletAllocation(n_components=optimal_k_essential, learning_method='online', n_jobs=-1, random_state=42)
topics_essential = lda_essential.fit_transform(X_essential)

df.loc[df['Versiontype'] == 'Essential', 'Essential_Topic'] = topics_essential.argmax(axis=1)

reducer_subscription = umap.UMAP(n_neighbors=15, n_components=10, random_state=42)
X_umap_subscription = reducer_subscription.fit_transform(X_subscription.toarray())

clusterer_subscription = hdbscan.HDBSCAN(min_cluster_size=20, min_samples=5)
labels_subscription = clusterer_subscription.fit_predict(X_umap_subscription)
optimal_k_subscription = len(set(labels_subscription)) - (1 if -1 in labels_subscription else 0)
optimal_k_subscription = min(optimal_k_subscription, 10)
print(f"Subscription最佳主題數量（n_components）: {optimal_k_subscription}")

lda_subscription = LatentDirichletAllocation(n_components=optimal_k_subscription, learning_method='online', n_jobs=-1, random_state=42)
topics_subscription = lda_subscription.fit_transform(X_subscription)

df.loc[df['Versiontype'] == 'Subscription', 'Subscription_Topic'] = topics_subscription.argmax(axis=1)

terms = vectorizer.get_feature_names_out()

def print_keywords(lda_model, n_words=10):
    """列出每個主題的關鍵詞"""
    for topic_idx, topic in enumerate(lda_model.components_):
        print(f"\nTopic #{topic_idx}:")
        print(" ".join([terms[i] for i in topic.argsort()[:-n_words - 1:-1]]))

print("\nEssential 類別的主題關鍵詞：")
print_keywords(lda_essential)

print("\nSubscription 類別的主題關鍵詞：")
print_keywords(lda_subscription)

topic_counts_essential = df[df['Versiontype'] == 'Essential']['Essential_Topic'].value_counts().sort_index()
print("\nEssential 類別每個主題的問題數量：")
print(topic_counts_essential)

topic_counts_subscription = df[df['Versiontype'] == 'Subscription']['Subscription_Topic'].value_counts().sort_index()
print("\nSubscription 類別每個主題的問題數量：")
print(topic_counts_subscription)
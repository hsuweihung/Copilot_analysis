import pandas as pd
import numpy as np
import umap.umap_ as umap
import hdbscan
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.metrics.pairwise import cosine_similarity

df = pd.read_excel(
    "/Users/williamhsu/Desktop/Copilot_analysis/Copilot_analysis/Copilot_unhelp.xlsx")

df = df.dropna(subset=['Prompt', 'Copilot Reply'])

essential_prompts = df[df['Versiontype'] == 'Essential']['Prompt'].values
subscription_prompts = df[df['Versiontype'] == 'Subscription']['Prompt'].values

count_vectorizer = CountVectorizer(max_df=0.9, min_df=5, stop_words='english')
tfidf_vectorizer = TfidfVectorizer(stop_words='english')

all_prompts = np.concatenate([essential_prompts, subscription_prompts])
X_all = count_vectorizer.fit_transform(all_prompts)

X_essential = X_all[:len(essential_prompts)]
X_subscription = X_all[len(essential_prompts):]


def process_topic_modeling(X, version_type, original_length):
    """對 Essential 或 Subscription 進行主題建模"""

    reducer = umap.UMAP(n_neighbors=10, n_components=5, random_state=42)
    X_umap = reducer.fit_transform(X.toarray())

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=5, min_samples=2, cluster_selection_method='eom')
    labels = clusterer.fit_predict(X_umap)

    unique_labels = set(labels)
    print(f"{version_type} 群集標籤: {unique_labels}")

    optimal_k = max(2, min(len(unique_labels) -
                    (1 if -1 in labels else 0), 10))

    if optimal_k == 2 and -1 in labels:
        print(f"⚠️ {version_type} 無法找到足夠的有效群集，將使用預設主題數量 5")
        optimal_k = 5

    print(f"{version_type} 最佳主題數量: {optimal_k}")

    lda = LatentDirichletAllocation(
        n_components=optimal_k, learning_method='online', random_state=42)
    valid_idx = X.toarray().sum(axis=1) > 0
    topics = np.zeros((original_length, optimal_k))
    topics[valid_idx] = lda.fit_transform(X[valid_idx])

    return topics, lda, optimal_k


topics_essential, lda_essential, _ = process_topic_modeling(
    X_essential, "Essential", len(essential_prompts))
df.loc[df['Versiontype'] == 'Essential',
       'Essential_Topic'] = topics_essential.argmax(axis=1)

topics_subscription, lda_subscription, _ = process_topic_modeling(
    X_subscription, "Subscription", len(subscription_prompts))
df.loc[df['Versiontype'] == 'Subscription',
       'Subscription_Topic'] = topics_subscription.argmax(axis=1)

terms = count_vectorizer.get_feature_names_out()


def print_keywords(lda_model, version_type, n_words=10):
    print(f"\n{version_type} 主題關鍵詞：")
    for topic_idx, topic in enumerate(lda_model.components_):
        print(f"Topic #{topic_idx}: " +
              " ".join([terms[i] for i in topic.argsort()[:-n_words - 1:-1]]))


print_keywords(lda_essential, "Essential")
print_keywords(lda_subscription, "Subscription")

topic_counts_essential = df[df['Versiontype'] ==
                            'Essential']['Essential_Topic'].value_counts().sort_index()
print("\nEssential 類別每個主題的問題數量：")
print(topic_counts_essential)

topic_counts_subscription = df[df['Versiontype'] ==
                               'Subscription']['Subscription_Topic'].value_counts().sort_index()
print("\nSubscription 類別每個主題的問題數量：")
print(topic_counts_subscription)

tfidf_matrix = tfidf_vectorizer.fit_transform(
    df['Prompt'].tolist() + df['Copilot Reply'].tolist())
tfidf_prompt = tfidf_matrix[:len(df)]
tfidf_response = tfidf_matrix[len(df):]
df['Cosine_Similarity'] = [cosine_similarity(tfidf_prompt[i], tfidf_response[i])[
    0][0] for i in range(len(df))]

df_high_sim = df[df['Cosine_Similarity'] > 0.8]
print("\n=== 高相似度但仍 Unhelp 的案例 ===")
print(df_high_sim[['Prompt', 'Copilot Reply', 'Cosine_Similarity']])

df['Response_Length'] = df['Copilot Reply'].apply(lambda x: len(x.split()))
df['Response_Length_Group'] = pd.cut(df['Response_Length'], bins=[
                                     0, 10, 30, 60, 100, np.inf], labels=['<10', '10-30', '30-60', '60-100', '100+'])

print("\n=== 不同回應長度的分布 ===")
print(df['Response_Length_Group'].value_counts())

negative_keywords = ['sorry', 'no information',
                     'not sure', 'unavailable', 'i don’t know']
df['Has_Negative_Words'] = df['Copilot Reply'].apply(
    lambda x: any(word in x.lower() for word in negative_keywords))

print("\n=== 包含負面詞的回應數量 ===")
print(df['Has_Negative_Words'].value_counts())


def categorize_unhelp(row):
    """判斷 Unhelp 可能的原因"""
    if row['Cosine_Similarity'] > 0.8:
        return "重複問題"
    elif row['Response_Length'] < 10:
        return "回應太短"
    elif row['Response_Length'] > 100:
        return "回應太長"
    elif row['Has_Negative_Words']:
        return "負面回應"
    return "其他"


df['Unhelp_Type'] = df.apply(categorize_unhelp, axis=1)

print("\n=== Unhelp 主要原因分佈 ===")
print(df['Unhelp_Type'].value_counts())

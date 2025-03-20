import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation


df = pd.read_excel("C:\\Users\\William_Hsu\\Desktop\\Copilot_answer.xlsx")
prompts = df['Prompt'].values

All_user_prompts = df[df['Versiontype'] == 'All_user']['Prompt'].values

vectorizer = CountVectorizer()

X_All_user = vectorizer.fit_transform(All_user_prompts)

lda_All_user = LatentDirichletAllocation(n_components=10, random_state=42)
lda_All_user.fit(X_All_user)

topics_All_user = lda_All_user.transform(X_All_user)

df.loc[df['Versiontype'] == 'All_user', 'All_user_Topic'] = topics_All_user.argmax(axis=1)

terms = vectorizer.get_feature_names_out()

def print_keywords(lda_model, n_words=10):
    """列出每個主題的關鍵詞"""
    for topic_idx, topic in enumerate(lda_model.components_):
        print(f"\nTopic #{topic_idx}:")
        print(" ".join([terms[i] for i in topic.argsort()[:-n_words - 1:-1]]))

print("All_user 類別的主題關鍵詞：")
print_keywords(lda_All_user)

topic_counts = df[df['Versiontype'] == 'All_user']['All_user_Topic'].value_counts().sort_index()
print("\nAll_user 類別每個主題的問題數量：")
print(topic_counts)



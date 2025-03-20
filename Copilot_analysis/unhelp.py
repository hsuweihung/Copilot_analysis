import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

df = pd.read_excel("C:\\Users\\William_Hsu\\Desktop\\AI Copilot Un-Helpful Prompt and Reply.xlsx")
prompts = df['Prompt_en_lower'].values

essential_prompts = df[df['Versiontype'] == 'Essential']['Prompt_en_lower'].values
subscription_prompts = df[df['Versiontype'] == 'Subscription']['Prompt_en_lower'].values

vectorizer = CountVectorizer(stop_words='english')

X_essential = vectorizer.fit_transform(essential_prompts)
lda_essential = LatentDirichletAllocation(n_components=10, random_state=42)
lda_essential.fit(X_essential)
topics_essential = lda_essential.transform(X_essential)
df.loc[df['Versiontype'] == 'Essential', 'Essential_Topic'] = topics_essential.argmax(axis=1)

X_subscription = vectorizer.fit_transform(subscription_prompts)
lda_subscription = LatentDirichletAllocation(n_components=5, random_state=42)
lda_subscription.fit(X_subscription)
topics_subscription = lda_subscription.transform(X_subscription)
df.loc[df['Versiontype'] == 'Subscription', 'Subscription_Topic'] = topics_subscription.argmax(axis=1)


print("Essential類別的問題：")
print(df[df['Versiontype'] == 'Essential'][['Prompt_en_lower', 'Essential_Topic']].head())

print("Subscription類別的問題：")
print(df[df['Versiontype'] == 'Subscription'][['Prompt_en_lower', 'Subscription_Topic']].head())

terms = vectorizer.get_feature_names_out()

def print_keywords(lda_model, n_words=5):
    for topic_idx, topic in enumerate(lda_model.components_):
        print(f"\nTopic #{topic_idx}:")
        print(" ".join([terms[i] for i in topic.argsort()[:-n_words - 1:-1]]))

print("Essential類別的主題關鍵詞：")
print_keywords(lda_essential)

print("Subscription類別的主題關鍵詞：")
print_keywords(lda_subscription)


print("\nEssential類別每個問題的主題分佈：")
for i, row in df[df['Versiontype'] == 'Essential'].iterrows():
    topic_idx = row['Essential_Topic']
    print(f"Prompt: {row['Prompt_en_lower']} -> Assigned Topic: {topic_idx}")

print("\nSubscription類別每個問題的主題分佈：")
for i, row in df[df['Versiontype'] == 'Subscription'].iterrows():
    topic_idx = row['Subscription_Topic']
    print(f"Prompt: {row['Prompt_en_lower']} -> Assigned Topic: {topic_idx}")

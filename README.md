# OPINION-ANALYSIS-BY-USING-ARTIFICIAL-INTELLIGENCE-IN-SOCIAL-MEDIA
import numpy as np 
import pandas as pd 
import re
import matplotlib.pyplot as plt
%matplotlib inline


airline_tweets = pd.read_csv('Tweets.csv')
airline_tweets.head()


plot_size = plt.rcParams["figure.figsize"] 
print(plot_size[0])
print(plot_size[1])
plot_size[0] = 8
plot_size[1] = 6
plt.rcParams["figure.figsize"] = plot_size 



airline_tweets.airline.value_counts().plot(kind='pie', autopct='%1.0f%%')


airline_tweets.airline_sentiment.value_counts().plot(kind='pie', autopct='%1.0f%%', colors=["red", "yellow", "green"])


airline_sentiment = airline_tweets.groupby(['airline', 'airline_sentiment']).airline_sentiment.count().unstack()
airline_sentiment.plot(kind='bar')



import seaborn as sns
sns.barplot(x='airline_sentiment', y='airline_sentiment_confidence' , data=airline_tweets)



features = airline_tweets.iloc[:, 10].values
labels = airline_tweets.iloc[:, 1].values




processed_features = []
for sentence in range(0, len(features)):
    # Remove all the special characters
    processed_feature = re.sub(r'\W', ' ', str(features[sentence]))
    # remove all single characters
    processed_feature= re.sub(r'\s+[a-zA-Z]\s+', ' ', processed_feature)
    # Remove single characters from the start
    processed_feature = re.sub(r'\^[a-zA-Z]\s+', ' ', processed_feature) 
    # Substituting multiple spaces with single space
    processed_feature = re.sub(r'\s+', ' ', processed_feature, flags=re.I)
    # Removing prefixed 'b'
    processed_feature = re.sub(r'^b\s+', '', processed_feature)
    # Converting to Lowercase
    processed_feature = processed_feature.lower()
    processed_features.append(processed_feature)



from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(processed_features, labels, test_size=0.2, random_state=0)



from google.colab import drive
drive.mount('/content/drive')

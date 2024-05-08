#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Import required libraries
import newspaper
import matplotlib.pyplot as plt
import pandas as pd
import nltk
import numpy as np
import networkx as nx
import re
import string
import torch
import tensorflow as tf
from nltk.corpus import stopwords
from nltk.cluster.util import cosine_distance
from nltk.tokenize import sent_tokenize
from newspaper import Article
from pygooglenews import GoogleNews
from newspaper import ArticleException
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
from PIL import Image
from textblob import TextBlob
from transformers import pipeline
from scipy.spatial.distance import cosine


# In[2]:


# create an empty array to store articles
news = [] 


# In[3]:


# Create an instance of the GoogleNews class
gn = GoogleNews()


# In[4]:


# Set the desired search query
search_query = "ASLEF Strike"


# In[5]:


# Fetch top 20 news articles
search_results = gn.search(search_query)


# In[6]:


# Counter to keep track of the number of articles processed
articles_processed = 0
 
# Iterate through the search results
for entry in search_results['entries']:
    try:
        # Get the URL of the article
        article_url = entry['link']
       
        # Download and parse the article using newspaper3k
        article = Article(article_url)
        article.download()
        article.parse()
       
        # Print the title, link, and body of the article
        print("Title:", article.title)
        print("Link:", article_url)
        print("Body:", article.text)
        print("\n")
               
        news.append([article.title,article_url,article.text])
    
        # Increment the counter
        articles_processed += 1
              
        # Break the loop if we have processed 20 articles
        if articles_processed >= 20:
            break
   
    except ArticleException as e:
        # Handle the exception (e.g., log the error, skip the article)
        #print("Error")
        print(f"Error processing article: {str(e)}")
        continue


# In[7]:


# Create Pandas Dataframe containing the array of news articles
df = pd.DataFrame(news)


# In[8]:


# Rename dataframe columns
df=df.rename(columns={0: "Title", 1: "URL",2:"Body"})


# In[9]:


# View dataframe
df


# ## Word Cloud

# In[10]:


# Create word cloud of first news article
text = df.iloc[0][2]

# Generate word cloud
wc = WordCloud().generate(text)

# Display the word cloud
plt.figure(figsize=(10, 8))
plt.imshow(wc, interpolation='bilinear')
plt.axis("off")
plt.show()


# ## Sentence Length Distribution

# In[11]:


sentence_lengths = df["Body"].apply(lambda x: [len(sent.split()) for sent in sent_tokenize(x)])

# Flatten the list of sentence lengths
all_sentence_lengths = [length for lengths in sentence_lengths for length in lengths]

# Plot the distribution of sentence lengths
plt.figure(figsize=(10, 6))
plt.hist(all_sentence_lengths, bins=30, color='skyblue', edgecolor='black')
plt.title('Distribution of Sentence Lengths')
plt.xlabel('Number of Words in Sentence')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()


# ## Article Length Distribution

# In[12]:


# Tokenize each article into words and calculate article lengths
article_lengths = df["Body"].apply(lambda x: len(x.split()))

# Plot the distribution of article lengths
plt.figure(figsize=(10, 6))
plt.hist(article_lengths, bins=30, color='skyblue', edgecolor='black')
plt.title('Distribution of Article Lengths')
plt.xlabel('Number of Words in Article')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()


# ## Sentiment Analysis

# In[13]:


# Function to calculate sentiment polarity of a text using TextBlob
def calculate_sentiment(text):
    blob = TextBlob(text)
    return blob.sentiment.polarity

# Apply the function to each article in the column df["Body"]
df["Sentiment"] = df["Body"].apply(calculate_sentiment)

# Print the first few rows of the DataFrame with sentiment scores
print(df[["Body", "Sentiment"]].head())


# In[14]:


# Plot the distribution of sentiment scores
plt.figure(figsize=(10, 6))
plt.hist(df["Sentiment"], bins=30, color='skyblue', edgecolor='black')
plt.title('Distribution of Sentiment Scores')
plt.xlabel('Sentiment Score')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()


# In[15]:


# Function to calculate sentiment polarity of a text using TextBlob
def calculate_subjectivity(text):
    blob = TextBlob(text)
    return blob.sentiment.subjectivity

# Apply the function to each article in the column df["Body"]
df["Subjectivity"] = df["Body"].apply(calculate_subjectivity)

# Print the first few rows of the DataFrame with sentiment scores
print(df[["Body", "Subjectivity"]].head())


# In[16]:


# Plot the distribution of sentiment scores
plt.figure(figsize=(10, 6))
plt.hist(df["Subjectivity"], bins=30, color='skyblue', edgecolor='black')
plt.title('Distribution of Subjectivity Scores')
plt.xlabel('Subjectivity Score')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()


# ## Extractive Text Summarisation Method

# ### Lowercasing

# In[17]:


# Lowercase all letter in the body of articles
df['Body'] = df['Body'].str.lower()
df


# ### Sentence Tokenisation

# In[18]:


def read_article(text):
    sentences = sent_tokenize(text)
    for i, sentence in enumerate(sentences):
        sentences[i] = re.sub(r'[^a-zA-Z0-9]', ' ', sentence)
    return sentences


# In[19]:


def sentence_similarity(sent1, sent2, stopwords=None):
    if stopwords is None:
        stopwords = []
    sent1 = [w.lower() for w in sent1]
    sent2 = [w.lower() for w in sent2]

    all_words = list(set(sent1 + sent2))

    vector1 = [0] * len(all_words)
    vector2 = [0] * len(all_words)
    
    # Build the vector for the first sentence
    for w in sent1:
        if w not in stopwords:
            vector1[all_words.index(w)] += 1
    # Build the vector for the second sentence
    for w in sent2:
        if w not in stopwords:
            vector2[all_words.index(w)] += 1
            
    return 1 - cosine(vector1, vector2)


# In[20]:


def build_similarity_matrix(sentences, stop_words):
    # Create an empty similarity matrix
    similarity_matrix = np.zeros((len(sentences), len(sentences)))

    # Iterate over each pair of sentences
    for idx1 in range(len(sentences)):
        for idx2 in range(len(sentences)):
            if idx1 != idx2:
                similarity_matrix[idx1][idx2] = sentence_similarity(sentences[idx1], sentences[idx2], stop_words)
                
    return similarity_matrix


# In[21]:


def generate_summary(text, top_n):
    nltk.download('stopwords')    
    nltk.download('punkt')
    stop_words = stopwords.words('english')    
    summarize_text = []
    # Step 1: Tokenize the text into sentences
    sentences = read_article(text)
    # Step 2: Generate the similarity matrix
    sentence_similarity_matrix = build_similarity_matrix(sentences, stop_words)
    # Step 3: Rank sentences in the similarity matrix
    sentence_similarity_graph = nx.from_numpy_array(sentence_similarity_matrix)
    scores = nx.pagerank(sentence_similarity_graph)
    # Step 4: Sort the ranks and select top sentences
    ranked_sentences = sorted(((scores[i], s) for i, s in enumerate(sentences)), reverse=True)
    # Step 5: Get the top n sentences based on rank
    for i in range(top_n):
        summarize_text.append(ranked_sentences[i][1])
    # Step 6: Output the summarized version
    return " ".join(summarize_text), len(sentences)


# In[22]:


# Original Article
print("Original Article: ", df["Body"][0])


# In[23]:


# Produce Extractive summary of article
extractive=generate_summary(df["Body"][0],3)
print("Extractive summary: ", extractive)


# In[24]:


# Generate Extractive Word Cloud
wc = WordCloud().generate(extractive[0])

# Display the word cloud
plt.figure(figsize=(10, 8))
plt.imshow(wc, interpolation='bilinear')
plt.axis("off")
plt.show()


# In[25]:


# Length of orginal article
len(df["Body"][0])


# In[26]:


# Length of summarised version through extractive method
len(extractive[0])


# ## Abstractive Text Summarisation Model (BART)

# In[27]:


# Redefine dataframe
df = pd.DataFrame(news)


# In[28]:


# Define column titles again
df=df.rename(columns={0: "Title", 1: "URL",2:"Body"})


# In[29]:


# Filter out articles that are too long or too short for the model to work
df["NBody"]=df["Body"].str.split()
df["WordsCount"]=df["NBody"].apply(lambda x: len(x)) 
df=df[(df["WordsCount"]>= 130)& (df["WordsCount"]<= 1024)]


# In[30]:


# Define a function to remove punctuation
def remove_punctuation(text):
    return text.translate(str.maketrans('', '', string.punctuation))

# Apply the function to the 'text' column
df['Body'] = df['Body'].apply(remove_punctuation)


# In[31]:


df.reset_index(drop=True,inplace=True)


# In[ ]:


# Load the summarization pipeline with the BART model
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

# Initialize an empty list to store the final summaries
all_summaries = []

# Iterate over each string in the "Body" column of the DataFrame
for idx, input_text in enumerate(df["Body"]):
    # Generate the summary for the current input text
    summary = summarizer(input_text, max_length=130, min_length=30, do_sample=False)
    # Append the summary text to the list of final summaries
    all_summaries.append(summary[0]['summary_text'])

    # Print the news article title and summary
    print(f"\033[1mSummary for news article {idx + 1}: {df['Title'][idx]}\033[0m\n{summary[0]['summary_text']}\n")


# In[ ]:


# Length of first abstractive summarised article
len(all_summaries[0])


# ## WordCloud

# In[ ]:


# Generate word cloud
wc = WordCloud().generate(summary[0]['summary_text'])

# Display the word cloud
plt.figure(figsize=(10, 8))
plt.imshow(wc, interpolation='bilinear')
plt.axis("off")
plt.show()


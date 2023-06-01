#!/usr/bin/env python
# coding: utf-8

# In[2]:


import nltk
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('omw-1.4')
import csv
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import string
import pandas as pd
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.corpus import wordnet

def process_block(block):
    # Удаляем все специальные символы и цифры
    block = ''.join([i for i in block if not i.isdigit() and i not in string.punctuation])
    # Токенизация 
    tokens = nltk.word_tokenize(block.lower())
    # Лемматизация
    lemmatizer = WordNetLemmatizer()
    lemmas = [lemmatizer.lemmatize(token) for token in tokens]
    # Удаление стоп-слов
    stop_words = set(stopwords.words('english'))
    keywords = [lemma for lemma in lemmas if lemma not in stop_words]
    # Возвращаем список ключевых слов
    return keywords

def mapper(file_path, block_size):
    with open(file_path) as f:
        reader = csv.reader(f)
        # Читаем первый блок
        block = ''
        for line in reader:
            block += ' '.join(line) + ' '
            if len(block) > block_size:
                # Очищаем и обрабатываем блок
                keywords = process_block(block)
                # Возвращаем ключевые слова
                yield keywords
                # Сбрасываем блок
                block = ''
        # Обработка последнего блока
        if len(block) > 0:
            keywords = process_block(block)
            yield keywords

def reducer(keywords):
    word_counts = {}
    for keyword in keywords:
        if keyword in word_counts:
            word_counts[keyword] += 1
        else:
            word_counts[keyword] = 1
    return word_counts

file_path = 'testov.csv'
block_size = 10000

# Вызываем mapper
mapped_data = []
for keywords in mapper(file_path, block_size):
    mapped_data.append(keywords)

# Вызываем reducer
word_counts = {}
for keywords in mapped_data:
    for word, count in reducer(keywords).items():
        if word in word_counts:
            word_counts[word] += count
        else:
            word_counts[word] = count

# Создаем два списка для существующих и несуществующих слов
existing_words = []
non_existing_words = []

# Проходим по всем словам в словаре
for word, count in sorted(word_counts.items(), key=lambda x: x[1], reverse=True):
    synsets = []
    for pos in ['n', 'v', 'a', 'r']:
        synsets.extend(wordnet.synsets(word, pos=pos))
    if len(synsets) > 0:
        existing_words.append({'word': word, 'count': count})
    else:
        non_existing_words.append({'word': word, 'count': count})

# Сохраняем списки в отдельные CSV файлы
df_existing = pd.DataFrame(existing_words)
df_non_existing = pd.DataFrame(non_existing_words)
df_existing.to_csv('existing_words.csv', index=False)
df_non_existing.to_csv('non_existing_words.csv', index=False)

# Выводим результаты в виде датафрейма
df_sorted = pd.DataFrame.from_dict(word_counts, orient='index', columns=['count'])
df_sorted.index.name = 'word'
df_sorted = df_sorted.reset_index().sort_values(by='count', ascending=False)

df_existing.head(30)


# In[4]:


existing_words


# In[6]:


from nltk.corpus import wordnet

book_synsets = list(wordnet.synsets('wa'))
print(book_synsets)


# In[ ]:





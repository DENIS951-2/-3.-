#!/usr/bin/env python
# coding: utf-8

# # Первоначальная версия датасета состоит из десяти столбцов, содержащих следующую информацию:
# 
# **Restaurant_id** — идентификационный номер ресторана / сети ресторанов;  
# **City** — город, в котором находится ресторан;  
# **Cuisine Style** — кухня или кухни, к которым можно отнести блюда, предлагаемые в ресторане;  
# **Ranking** — место, которое занимает данный ресторан среди всех ресторанов своего города;  
# **Rating** — рейтинг ресторана по данным TripAdvisor (именно это значение должна будет предсказывать модель);  
# **Price Range** — диапазон цен в ресторане;  
# **Number of Reviews** — количество отзывов о ресторане;  
# **Reviews** — данные о двух отзывах, которые отображаются на сайте ресторана;  
# **URL_TA** — URL страницы ресторана на TripAdvosor;  
# **ID_TA** — идентификатор ресторана в базе данных TripAdvisor  

# In[1]:


# Импортируем необходимые библиотеки:
import pandas as pd
import json 
from ast import literal_eval
import numpy as np
from pprint import pprint
import datetime
from datetime import datetime, timedelta
import re
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics 
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
import warnings; warnings.simplefilter('ignore')
sns.set()


# **Прочитаем датасет**

# In[2]:


df = pd.read_csv('main_task.csv')
df.head(3)


# **Заменим наименование столбцов для более удобной работы**

# In[3]:


df.columns = ['restaurant_id', 'city', 'cuisine_style', 'ranking',
              'rating', 'price_range', 'reviews_number', 'reviews', 'url_ta', 'id_ta']


# **Посмотрим на информацию о датафрейме**

# In[4]:


df.info()


# **Обнаружены пропуски в столбце Number of Reviews. Можно сделать предположение, что если в столбце reviews указан пустой список, значит количество отзывов = 0. Заменим такие пропуски.**

# In[5]:


df.loc[df.reviews_number.isna() & (df.reviews == '[[], []]'),
       'reviews_number'] = 0


# **Дальше мы можем предположить, что если в столбце reviews представлена информация только об одном отзыве, то количество отзывов = 1. Заменим такие пропуски.**

# In[6]:


df.loc[df.reviews_number.isna() & (df.reviews.str.match(
    '^\[\[\'.*\'\],\s?\[\'.*\'\]\]$')), 'reviews_number'] = 1


# **Проверим, что получилось**

# In[7]:


df.info()


# **Т.о. осталось только несколько пропусков в столбце reviews_number, заменим их медианой**

# In[8]:


df.fillna(value={'reviews_number': df.reviews_number.median()}, inplace=True)


# **Поскольку показатель ranking зависит от общего количества ресторанов в городе, то можно представить его в виде относительной величины как отношение к максимальному по городу**

# In[9]:


groups_by_cities = df.groupby('city').ranking.max()
df.ranking = df.apply(lambda x: x.ranking/groups_by_cities[x.city], axis=1)


# In[10]:


df['ranking'].hist(bins=10)


# **У нас есть категориальный признак cuisine_style, создадим на его основе dummy признаки. 
# При этом уменьшим количество значений этого признака, переимновав такие значения в Other**

# In[11]:


df_styles = df.cuisine_style.fillna('[]').apply(lambda x: literal_eval(x))
all_cuisine_styles = []
for style in df_styles:
    all_cuisine_styles += style
cuisine_styles = set(all_cuisine_styles)
cuisine_styles_new = set()
for style in cuisine_styles:
    freq = df.cuisine_style.str.contains(style).sum()
    if freq < 100:
        df.cuisine_style = df.cuisine_style.replace(style, 'Other')
        cuisine_styles_new.add('Other')
    else:
        cuisine_styles_new.add(style)
for style in cuisine_styles_new:
    df[style] = df.cuisine_style.fillna('').apply(
        lambda x: 1 if style in x else 0)


# In[12]:


df = df.join(pd.get_dummies(df.Other))


# **Создадим также dummy признаки для второго категориального признака - city**

# In[13]:


df = df.join(pd.get_dummies(df.city))


# **Посчитаем сколько дней прошло с момента последнего отзыва и добавим эти данные в качестве нового признака.
# При этом будем исходить из предположения, что датой составления датасета является максимальная дата отзыва
# из всего датасета. Также предположим, что если отзывов нет, эта цифра равна медианному количеству дней
# в датасете.**

# In[14]:


df.reviews = df.reviews.apply(lambda x: literal_eval(x.replace('nan', "''")))
df['review1_date'] = df.reviews.apply(lambda x: x[1][0] if len(x[1]) else '')
df['review2_date'] = df.reviews.apply(
    lambda x: x[1][1] if len(x[1]) > 1 else '')
df.review1_date = pd.to_datetime(df.review1_date)
df.review2_date = pd.to_datetime(df.review2_date)
df['last_review_date'] = df.apply(
    lambda x: x.review1_date if x.review1_date > x.review2_date else x.review2_date, axis=1)
last_date = df.last_review_date.max()
df['last_review_days'] = df.apply(lambda x: (
    last_date - x.last_review_date).days, axis=1)
df.last_review_days = df.last_review_days.fillna(df.last_review_days.median())


# **Приведем показатель price_range к числовому значению и заменим пропуски на медианное значение**

# In[15]:


df.price_range = df.price_range.replace('$', 1)
df.price_range = df.price_range.replace('$$ - $$$', 2)
df.price_range = df.price_range.replace('$$$$', 3)
df.price_range.fillna(df.price_range.median(), inplace=True)


# **Добавим новый признак, показывающий сколько дней прошло между двумя представленными в датасете отзывами
# и заполним максимальным значением те строки, где двух отзывов нет**

# In[16]:


df['interval'] = df.apply(lambda x: abs(
    (x.review1_date - x.review2_date).days), axis=1)
df.interval.fillna(df.interval.max(), inplace=True)


# **смотрим что получилось**

# In[17]:


df.info()


# **Посмотрим на распределение некоторых числовых признаков**

# In[18]:


df[['interval', 'last_review_days', 'reviews_number',
    'ranking', 'price_range']].hist(bins=5)


# **смотрим распределение по соотношению к ценовому диапазану**

# In[35]:


sns.pairplot(df_1, hue = 'price_range')


# **Удалим все нечисловые столбцы**

# In[19]:


df.drop(['review1_date', 'review2_date',
         'last_review_date'], axis=1, inplace=True)
for col in df:
    if df[col].dtype == np.object:
        df.drop(columns=[col], inplace=True)


# **Посмотрим на корреляцию признаков.**

# In[20]:


df[['rating', 'last_review_days', 'ranking',
    'reviews_number', 'interval', 'price_range']].corr()


# **Из таблицы мы видим, что силно скоррелированных признаков нет, а значит наш датасет пригоден
# для обучения модели**

# ## Разбиваем датафрейм на части, необходимые для обучения и тестирования модели, нормализация данных¶

# **Х - данные с информацией о ресторанах,
# у - целевая переменная (рейтинги ресторанов)
# Проведем нормализацию наших признаков**

# In[21]:


scaler = StandardScaler()
X = df.drop(['rating'], axis=1)
scaler.fit_transform(df)
y = df['rating']


#  **Наборы данных с меткой "train" будут использоваться для обучения модели, "test" - для тестирования.
# Для тестирования мы будем использовать 25% от исходного датасета.**

# In[22]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)


# **Создаём модель**
# 

# In[23]:


regr = RandomForestRegressor(n_estimators=100)


# **Обучаем модель на тестовом наборе данных**

# In[24]:


regr.fit(X_train, y_train)


# **Используем обученную модель для предсказания рейтинга ресторанов в тестовой выборке.
# Предсказанные значения записываем в переменную y_pred**

# In[26]:


y_pred = regr.predict(X_test)


# **Сравниваем предсказанные значения (y_pred) с реальными (y_test), и смотрим насколько они в среднем отличаются
#  Метрика называется Mean Absolute Error (MAE) и показывает среднее отклонение предсказанных значений от фактических.**

# In[27]:


print('MAE:', metrics.mean_absolute_error(y_test, y_pred))


# In[ ]:





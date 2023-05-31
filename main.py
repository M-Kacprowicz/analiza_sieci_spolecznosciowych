import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from operator import itemgetter
from networkx.algorithms.community.centrality import girvan_newman

import re
import networkx as nx
import matplotlib.pyplot as plt
from pyvis.network import Network
from operator import itemgetter
from matplotlib import cm
from nltk.corpus import stopwords
from nltk import bigrams
import collections

df = pd.read_csv("Twitter data.csv", index_col='Unnamed: 0')
df.drop(columns= ['_id', 'created_at', 'id', 'name', 'user_created', 'verified', 'preprocessed_text', 'token_count',
         'untokenized_text'], inplace=True)

df['mentions'] = df['text'].apply(lambda x: re.findall("@([a-zA-Z0-9_]{1,50})", str(x)))
print(df["mentions"].value_counts())


def get_mentions(list, index):
    try:
        return list[index]
    except:
        return None
    
    
df['first_mention'] = df['mentions'].apply(lambda x : get_mentions(x,0))
df['second_mention'] = df['mentions'].apply(lambda x : get_mentions(x,1))
df = df.loc[df['second_mention'].notna()]


data_mentions = df[['first_mention', 'second_mention', 'topic']]
siec = data_mentions[['first_mention', 'second_mention']].value_counts().to_frame().reset_index()
siec = siec.loc[siec[0] >= 50].rename(columns = {0: "Wystapienia"})





import streamlit as st
import base64
import joblib
import pandas as pd
import nltk
from nltk.corpus import stopwords
import re
import numpy as np
import os.path


import plotly.graph_objects as go

import time
start = time.process_time()

cat_dif = '''
In February 1848 the people of Paris rose in revolt against the constitutional monarchy of 
Louis-Philippe. Despite the existence of excellent narrative accounts, the February Days, 
as this revolt is called, have been largely ignored by social historians of the past two decades. 
For each of the three other major insurrections in nineteenth-century Paris—July 1830, June 1848,
and May 1871—there exists at least a sketch of participants’ backgrounds and an analysis, more or 
less rigorous, of the reasons for the occurrence of the uprisings. Only in the case of the February 
Revolution do we lack a useful description of participants that might characterize it in the light 
of what social history has taught us about the process of revolutionary mobilization.
'''

alice = '''
flowers and those cool fountains, but she could not even get her head though the doorway ;
“ and even if my head would go through,” thought poor Alice, “ it would be of very little
use without my shoulders. Oh, how I wish I could shut up like a telescope! I think I could,
if I only knew how to begin.” For, you see, so many out-of-the-way things had happened lately
that Alice had begun to think that very few things indeed were really impossible.
There seemed to be no use in waiting by the little door, so she went back to the table,
half hoping she might find another key on it, or at any rate a book of rules for shutting
people up like telescopes : this time she found a little bottle on it, (“which certainly was not
here before,” said Alice,) and tied round the neck of the bottle was a paper label with the
words “ DRINK ME” beautifully printed on it in large letters
'''

path_var=''

model = joblib.load(os.path.abspath(path_var+'CommonLit_Model.pkl'))
name_vectorizer = joblib.load(os.path.abspath(path_var+'Vectorizer.pkl'))

mms1 = joblib.load(os.path.abspath(path_var+'mms1.pkl'))
mms2 = joblib.load(os.path.abspath(path_var+'mms2.pkl'))

nwdf = pd.read_csv(os.path.abspath(path_var+'nwdf.csv'), low_memory=False)
# nwdf.rename(columns=nwdf.iloc[0], inplace = True)
# nwdf.drop([0], inplace = True)
nwdf.drop(index=333184, axis=0, inplace=True)
def change_scale_word_count(old_value):
    return ((float(old_value) - 12711) / (1551258643 - 12711)) * (1 - 0) + 0


nwdf['scaled_count'] = nwdf['count_occur'].apply(change_scale_word_count)
word_freq = dict(zip(nwdf.word, nwdf.scaled_count))


def get_score(excerpt):
    score = 0

    for i in excerpt.split(' '):
        try:
            score += word_freq[i]
        except KeyError:
            pass

    return score


def cleaner(excerpt):
    clean = nltk.word_tokenize(re.sub("[^a-zA-Z]", " ", excerpt).lower())
    clean = [word for word in clean if not word in set(stopwords.words("english"))]

    lem = nltk.WordNetLemmatizer()
    clean = [lem.lemmatize(word) for word in clean]
    return " ".join(clean)


def my_fun(text, length):
    tdf = pd.DataFrame(columns=['excerpt'])
    tdf = tdf.append({'excerpt': text[:length]}, ignore_index=True)
    tdf['ex_len'] = tdf.excerpt.apply(lambda x: len(x))
    tdf.excerpt = tdf.excerpt.apply(cleaner)
    tdf['excerpt_score'] = tdf.excerpt.apply(get_score)
    names_encoded = name_vectorizer.transform(tdf.excerpt)
    names_df = pd.DataFrame(data=names_encoded.toarray(), columns=name_vectorizer.get_feature_names())
    tdf = pd.concat([tdf, names_df], axis=1)
    tdf.drop(['excerpt'], axis=1, inplace=True)
    tdf.excerpt_score = mms2.transform(np.reshape(list(tdf.excerpt_score), (-1, 1)))
    ypred = model.predict(tdf)

    return 1 - ypred


def show_scores(name_text_list):
    length = 600
    names = []
    scores = []
    name_score = ''

    for i in name_text_list:
        res_score = round(float(my_fun(str(i[1]), length)), 3)
        names.append(i[0])
        scores.append(res_score)
        name_score += str(i[0])+' '+str(res_score) + ' | '

    return name_score, names, scores


def get_table_download_link(df):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    return f'<a href="data:file/csv;base64,{b64}" download="myfilename.csv">Download Results as CSV File</a>'

st.write('For each Excerpt, only first 600 characters matter !!')

titles = st.text_area("Titles separated by commas", 'Paste here')
texts = st.text_area("Excerpts separated by - #####", 'Paste Here')

inp_titles = []
inp_texts = []
predicted = False

if st.button("Predict"):

    inp_titles.extend(['Alice', 'CAT Difficult'])
    inp_texts.extend([alice, cat_dif])

    inp_titles.extend(titles.split(','))
    inp_texts.extend(texts.split('#####'))

    name_score, names, scores = show_scores(list(zip(inp_titles, inp_texts)))
    st.write('Time taken to predict: '+str(int(time.process_time() - start))+' seconds.')
    st.write(name_score)

    fig = go.Figure(data=[go.Bar(x=names, y=scores)])
    st.plotly_chart(fig)
    predicted = True

if predicted:
    pd.DataFrame(columns=['Names', 'Score'])
    df = pd.DataFrame({'Names': names, 'Scores': scores})
    st.markdown(get_table_download_link(df), unsafe_allow_html=True)







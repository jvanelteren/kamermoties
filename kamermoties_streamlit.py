#%%
import streamlit as st
import pandas as pd
from pathlib import Path
from streamlit.caching import cache
from top2vec import Top2Vec
import pickle
import numpy as np
import altair as alt
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA


st.title('Ik vind ... tracker')

st.markdown(
    """
    Wat vindt jij belangrijk? Deze app heeft met machine learning alle moties van afgelopen Tweede Kamerperiode geclusterd in 247 onderwerpen. De app matcht jouw zoekterm(en) aan gelijksoortige onderwerpen. Per onderwerp zie je:
    
    1. Hoeveel moties partijen hebben ingediend
    2. Hoe gelijkwaardig partijen hebben gestemd
    3. De moties die bij het onderwerp horen.

    Heb ook nog in twee blogposts over de moties gemaakt. 
    * [Deel 1](https://jvanelteren.github.io/blog/2021/02/20/kamermotiesEDA.html) over trends
    * [Deel 2](https://jvanelteren.github.io/blog/2021/03/10/kamermoties_topics.html) over de inhoud van de moties
    
    Veel plezier ermee! Jesse
    """)

@st.cache(allow_output_mutation=True)
def load_model():
    return Top2Vec.load("data/doc2vec_deep_bigram_enhanced_stopwords")
model = load_model()

@st.cache(allow_output_mutation=True)
def load_df():
    filename = 'data/df_including_topics_full.pickle'
    f = open(filename,"rb")
    return pickle.load(f)
df = load_df()



parties = ['VVD',
 'CDA',
 'ChristenUnie',
 'D66',
 'SGP',
 'FVD',
 'PVV',
 'PvdA',
 'DENK',
 'GroenLinks',
 'SP',
 'PvdD']

party_colors = {
  'CDA':'#5cb957',
  'ChristenUnie':'#00a5e8',
  'D66':'#04a438',
  'GroenLinks':'#48a641',
  'PVV':'#002759',
  'PvdA':'#df111a',
  'PvdD':'#006b2d',
  'SGP':'#d86120',
  'SP':'#e3001b',
  'VVD':'#ff7f0e',
  'DENK':'#17becf',
  'FVD':'#800000',
  'Groep Krol/vKA':'pink'}


def get_stem_column(largest):
    return [c for c in df.columns if 'Stem_' in c and c != 'Stem_persoon' and c[5:] in largest]
    
def get_pca(df, n_components=1, num_largest=None, return_ratio=False):
    largest = parties
    stem_column = get_stem_column(largest)
    source_year = df[stem_column].dropna(axis=1, how='all').T
    X_year = SimpleImputer(strategy='most_frequent').fit_transform(source_year)
    pca = PCA(n_components = n_components)
    pca = pca.fit(X_year)
    print('explained variance by factors', pca.explained_variance_ratio_,pca.explained_variance_ratio_.sum())  
    res_year = pca.transform(X_year)
    source = pd.DataFrame(res_year)
    source['partij'] = source_year.T.columns.str[5:]
    source = source.rename(index=str, columns={0: "x", 1: "y"}).sort_values('x',ascending=False)
    return (source, pca.explained_variance_ratio_) if return_ratio else source
#%%
def pca_topic(df, topic, kamer, twodim=False):
    size=800
    column_list = df.columns
    source = df[(df['Topic_initial'] == topic) & (df['Kamer'] == kamer)]
    num_moties = len(source)
    if twodim:
        source, explained_variance_ratio_ = get_pca(source, n_components = 2, return_ratio=True)
    else:
        source, explained_variance_ratio_ = get_pca(source, n_components = 1, return_ratio=True)
    mid = (source['x'].max() + source['x'].min())/2
    median = source['x'].median()
    if source[source['partij'] =='VVD']['x'].values > median: # make sure that VVD is on the right part of the x-axis
        source['x'] += 2 * (mid - source['x'])
    if twodim:
        points = alt.Chart(source).mark_point().encode(
        # x=alt.X('x:Q', axis=alt.Axis(title='Eerste factor')),
        # y=alt.Y('y:Q', axis=alt.Axis(title='Tweede factor')),
        x=alt.X('x:Q', axis=None),
        y=alt.Y('y:Q', axis=None),
        color=alt.Color("partij", scale = alt.Scale(domain=parties,range= [party_colors[p] for p in parties]), legend=None),
        tooltip=['partij:N']
        )

        text = points.mark_text(
            align='left',
            baseline='middle',
            dx=np.random.uniform(0,10),
            dy=np.random.uniform(0,10)
            # opacity=0.5
        ).encode(
            text='partij:N'
        ).properties(
            title=f'Stemgedrag op onderwerp {selected_topic}, {num_moties} moties, grafiek {round(sum(explained_variance_ratio_)*100)}% betrouwbaar'
        )


        chart = (points + text)
        chart.configure_axis(
            grid=False).configure_view(
            strokeWidth=1)
        
        return chart

search_term = st.text_input('Kies een woord of meerdere woorden', '')

if search_term != '':
    st.markdown(f'## Onderwerpen die het beste passen bij {search_term}')

    selected_topic = 'geen'

    # select topic
    try:
        topic_words, word_scores, topic_scores, topic_nums = model.search_topics(keywords= search_term.split() , num_topics=4)
        for i, topic in enumerate(topic_words):
            st.write('Onderwerp ', topic_nums[i], ' Match: ', round(topic_scores[i]*100), '%\n\n ',' '.join(word for word in topic[:20]))
            selected_topic = topic_nums[0]
    except:
        st.write('(Een van de) woorden komt niet voor in de ingediende moties. Probeer opnieuw')
    
    if selected_topic != 'geen':
        source = df[(df['Topic_initial'] == selected_topic) & (df['Kamer'] == 'Rutte III')]
        source = source.groupby(['Indienende_partij']).size().reset_index(name='Aantal moties')

        # Overview of topic distribution over all years
        chart = alt.Chart(source).mark_bar().encode(
            x=alt.X('Indienende_partij:O', sort='-y'),
            y=alt.Y('Aantal moties:Q')
            # sort=alt.EncodingSortField('Aantal moties', order='descending'))
            # order=alt.Order('Aantal moties:Q',sort='descending')
        )
        st.markdown(f'## Moties ingediend per politieke partij op dit onderwerp')
        st.altair_chart(chart, use_container_width=True)
        # width and height does not work altair/streamlit
        st.markdown(f'## Stemgedrag van partijen op dit onderwerp')
        st.altair_chart(pca_topic(df, selected_topic, 'Rutte III', twodim=True), use_container_width=True)

        #%%
        # years = df["year"].loc[df["make"] = make_choice]
        # year_choice = st.sidebar.selectbox('', years) 

        def find_semantically_similar_docs(search_term):
            documents, document_scores, document_ids = model.search_documents_by_topic(topic_num =selected_topic, num_docs=5)
            # documents, document_scores, document_ids = model.search_documents_by_keywords(keywords=search_term.split(), num_docs=3)
            for doc, score, doc_id in zip(documents, document_scores, document_ids):
                st.markdown(f"Ingediend door {df.iloc[doc_id]['Indienende_persoon_partij']}, resultaat {df.iloc[doc_id]['BesluitTekst']}")
                st.markdown(f"Voor: {', '.join(df.iloc[doc_id]['Partijen_Voor'])}")
                st.markdown(f"Tegen: {', '.join(df.iloc[doc_id]['Partijen_Tegen'])}")
                st.text_area('Inhoud van de motie', doc, height=500, key=doc_id)

                print(f"Document: {doc_id}, Score: {score}")

        st.markdown('## Moties die het beste bij het onderwerp passen')
        find_semantically_similar_docs(search_term)

        partij_choice = st.sidebar.selectbox('Select your vehicle:', [1,2,3,4])
        with st.beta_expander("⚙️ - Gedetailleerde uitleg ", expanded=False):
            st.write(
                """    
            - Paste a Wikipedia URL.
            - Make sure the URL belongs to https://en.wikipedia.org/
                """
            )
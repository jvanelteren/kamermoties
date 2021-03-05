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
from copy import deepcopy
st.title('Ik vind ... tracker')

st.markdown(
    """
    Wat vindt jij belangrijk bij de verkiezingen? Deze app heeft met machine learning alle moties van afgelopen Tweede Kamerperiode geclusterd in 247 onderwerpen. De app matcht jouw zoekterm(en) aan gelijksoortige onderwerpen. Per onderwerp zie je:
    
    1. Hoeveel moties partijen hebben ingediend
    2. Hoe partijen hebben gestemd
    3. Welke moties bij jouw onderwerp horen.

    Heb ook nog in twee blogposts over de moties gemaakt. 
    * [Deel 1](https://jvanelteren.github.io/blog/2021/02/20/kamermotiesEDA.html) over trends
    * [Deel 2](https://jvanelteren.github.io/blog/2021/03/10/kamermoties_topics.html) over de inhoud van de moties

    Veel plezier ermee en succes met stemmen 17 maart!
    """)

@st.cache(allow_output_mutation=True)
def load_model():
    return Top2Vec.load("data/doc2vec_deep_bigram_enhanced_stopwords")

@st.cache(allow_output_mutation=True)
def load_df():
    filename = 'data/df_including_topics_full.pickle'
    with open(filename,"rb") as f:
        return pickle.load(f)



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

def get_df_slice(df):
        source = df[(df['Topic_initial'] == selected_topic) & (df['Kamer'] == 'Rutte III')]
        if per_partij:
            source = source[source['Indienende_partij'] == selected_party]
        if selected_soort == 'Aangenomen':
            source = source[source['BesluitSoort'] == 1]
        if selected_soort == 'Verworpen':
            source = source[source['BesluitSoort'] == 0]
        if len(source)==0:
            error = True
        return source

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
        color=alt.Color("partij", scale = alt.Scale(domain=parties,range= [party_colors[p] for p in parties]), legend=None)
        )

        text = points.mark_text(
            align='left',
            baseline='middle',
            size=26,
            dx=np.random.uniform(5,20),
            dy=np.random.uniform(5,20)
            # opacity=0.5
        ).encode(
            text='partij:N'
        )

        chart = (points + text)
        chart.configure_axis(
            labelFontSize=14,
            titleFontSize=14,
            grid=False).configure_view(
            strokeWidth=0).configure_title(fontSize=66)
        
        st.write(f'Stemgedrag op onderwerp {selected_topic}, {num_moties} moties, grafiek {round(sum(explained_variance_ratio_)*100)}% betrouwbaar')
        
        with st.beta_expander("⚙️ - Uitleg ", expanded=False):
            st.write(
                """    
            Deze tecniek heet Pricipal Component Analysis en probeert variatie op veel dimensies (in dit geval veel moties)
            terug te brengen naar minder dimensies (in dit geval twee, een x en een y as). Als je bijvoorbeeld twee partijen hebt die
            altijd precies tegenovergesteld stemmen dan heb hoef je niet heel veel verschillende moties te visualiseren, maar kan je gewoon de twee tegenover elkaar
            op één as tekenen.

            Het betrouwbaarheid percentage geeft aan hoeveel van de variatie in het stemgedrag wordt verklaard door de grafiek. Hoe lager dit is des te minder waarde 
            kan je er aan hechten.
                """
            )
        return chart

def aantal_moties_chart(df):
    # Overview of topic distribution over all years
    return alt.Chart(df).mark_bar().encode(
        x=alt.X('Indienende_partij:O', sort='-y',title=None),
        y=alt.Y('Aantal moties:Q',title=None)
        # sort=alt.EncodingSortField('Aantal moties', order='descending'))
        # order=alt.Order('Aantal moties:Q',sort='descending')
    ).configure_axis(
            labelFontSize=14,
            titleFontSize=14,
            grid=False).configure_view(
            strokeWidth=0)

model = load_model()
df = load_df()
@st.cache
def get_num2size():
    topic_sizes, topic_nums = model.get_topic_sizes()
    return {int(num):int(size) for num, size in zip(topic_nums, topic_sizes)}
num2size = get_num2size()

search_term = st.text_input('Kies je zoekterm(en)', '')

# select relevant topic topic
if search_term != '':
    try:
        topic_words, word_scores, topic_scores, topic_nums = model.search_topics(keywords= search_term.split() , num_topics=4)
        # selected_topic = topic_nums[0]
        error = False
    except:
        st.write('(Een van de) woorden komt niet voor in de ingediende moties. Probeer opnieuw')
        error = True

    if not error:
        st.markdown(f'## Onderwerpen die het beste passen bij {search_term}')
        with st.beta_expander("⚙️ - Uitleg ", expanded=False):
            st.write(
                """    
            Het Top2Vec algoritme heeft op basis van de woorden in de moties ze geclustert in heel veel onderwerpen.
            Alle onderwerpen hebben een nummer gekregen, beginnend met 0. Het ontwerp dat het beste matched met jouw
            zoekterm staat bovenaan. Een match score van boven de 20 is meestal wel goed.

            De woorden die vervolgens worden weergegeven zijn de woorden die volgens het model het meest onderscheidend zijn voor dit onderwerp
            Lees de woorden door dan krijg je een idee wat er met het onderwerp ongeveer bedoelt wordt. Vervolgens kan je ook nog met de filters
            de andere onderwerpen kiezen om deze verder te onderzoeken.
                """
            )
        for i, topic in enumerate(topic_words):
            st.write('Onderwerp ', topic_nums[i], ' Match: ', round(topic_scores[i]*100), '\n\n ',' '.join(word for word in topic[:20]))

        st.sidebar.markdown('Gebruik deze filters om de moties verder te filteren. De grafieken en moties updaten vanzelf')
        selected_topic = st.sidebar.radio("Kies je onderwerp: ", (topic_nums), key=1)
        selected_soort = st.sidebar.radio("Wat voor soort moties: ", (['Maakt niet uit','Aangenomen', 'Verworpen']), key=3)
        per_partij = st.sidebar.checkbox('Per partij filteren?')
        selected_party = st.sidebar.radio("Kies je partij: ", (sorted(parties)), key=2)
        skip_graphs = st.sidebar.checkbox('Schiet op (zet grafieken uit, voor als je alleen in de moties geïnteresseerd bent', value=False)
        max_moties = st.sidebar.slider('maximaal aantal weergegeven moties', 0, 20,5)

        # select data and plot charts
        source = get_df_slice(df)

        st.markdown(f'## {len(source)} Moties ingediend door partijen op onderwerp {selected_topic}')
        if not skip_graphs:
            chart = aantal_moties_chart(source.groupby(['Indienende_partij']).size().reset_index(name='Aantal moties'))
            st.altair_chart(chart, use_container_width=True)

            # width and height does not work altair/streamlit
            st.markdown(f'## Stemgedrag van partijen op onderwerp {selected_topic}')

            st.altair_chart(pca_topic(source, selected_topic, 'Rutte III', twodim=True), use_container_width=True)
            st.markdown(f'## Moties die het beste passen bij onderwerp {selected_topic}')


    


        documents, document_scores, document_ids = model.search_documents_by_topic(topic_num =selected_topic, num_docs= num2size[selected_topic])
        # documents, document_scores, document_ids = model.search_documents_by_keywords(keywords=search_term.split(), num_docs=3)
        count = 0
        for doc, score, doc_id in zip(documents, document_scores, document_ids):
            if count == max_moties: break
            if doc_id in list(source['Index']):
                summary = f"Ingediend door {df.iloc[doc_id]['Indienende_persoon_partij']}"
                result = f"Resultaat: {df.iloc[doc_id]['BesluitTekst']}"
                voor = f"Voor: {', '.join(df.iloc[doc_id]['Partijen_Voor'])}"
                tegen = f"Tegen: {', '.join(df.iloc[doc_id]['Partijen_Tegen'])}"
                st.write(summary, '  \n', result, '  \n', voor, '  \n', tegen)
                st.text_area('Inhoud van de motie:', doc, height=500, key=doc_id)
                # print(f"Document: {doc_id}, Score: {score}")
                count += 1


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
import os
import re


# code = """<script async defer data-domain="stemvinder.ew.r.appspot.com" src="https://plausible.io/js/plausible.js"></script>"""
# a=os.path.dirname(st.__file__)+'/static/index.html'
# with open(a, 'r') as f:
#     data=f.read()
#     if len(re.findall('plausible', data))==0:
#         with open(a, 'w') as ff:
#             newdata=re.sub('<head>','<head>'+code,data)
#             ff.write(newdata)






# hide hamburger menu
hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True) 







st.image('data/moties.jpg',use_column_width=True)
st.title('StemVinder')

st.markdown(
    """
    Wat vindt jij belangrijk bij de verkiezingen? Deze app heeft met machine learning alle moties van afgelopen Tweede Kamerperiode geclusterd in 247 onderwerpen. De app matcht jouw zoekterm(en) aan gelijksoortige onderwerpen. 
    Zo kan je er snel achterkomen hoe partijen hebben gestemd op wat jij belangrijk vindt. Per onderwerp zie je:
    
    1. Hoeveel moties partijen hebben ingediend
    2. Hoe partijen hebben gestemd
    3. Welke moties bij jouw onderwerp horen.

    Als je dit interessant vindt is er nog meer leesvoer:
    * [Blog 1](https://jvanelteren.github.io/blog/2021/02/20/kamermotiesEDA.html) over trends
    * [Blog 2](https://jvanelteren.github.io/blog/2021/03/07/kamermoties_topics.html) over de inhoud van de moties

    Veel plezier ermee en succes met stemmen 17 maart!
    """)

@st.cache(max_entries = 1, ttl = None, allow_output_mutation=True)
def load_model():
    return Top2Vec.load("data/doc2vec_production")

@st.cache(max_entries = 1, ttl = None, allow_output_mutation=True)
def load_df():
    filename = 'data/df_production.pickle'
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
    # print('explained variance by factors', pca.explained_variance_ratio_,pca.explained_variance_ratio_.sum())  
    res_year = pca.transform(X_year)
    source = pd.DataFrame(res_year)
    source['partij'] = source_year.T.columns.str[5:]
    source = source.rename(index=str, columns={0: "x", 1: "y"}).sort_values('x',ascending=False)
    return (source, pca.explained_variance_ratio_) if return_ratio else source

def get_df_slice(df):
        source = df[(df['Topic_initial'] == selected_topic) & (df['Kamer'] == 'Rutte III')]
        if selected_party != 'Alle partijen':
            source = source[source['Indienende_partij'] == selected_party]
        if selected_year != 'Alle jaren':
            source = source[source['Jaar'] == int(selected_year)]
        if selected_soort == 'Aangenomen':
            source = source[source['BesluitSoort'] == 'Aangenomen']
        if selected_soort == 'Verworpen':
            source = source[source['BesluitSoort'] == 'Verworpen']
        if len(source)==0:
            error = True
        return source

def pca_topic(df, topic, kamer, twodim=False):
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
        width = 700
        y_scale_ratio = explained_variance_ratio_[1]/explained_variance_ratio_[0]
        points = alt.Chart(source,width= width, height = width * y_scale_ratio).mark_point().encode(
        # x=alt.X('x:Q', axis=alt.Axis(title='Eerste factor')),
        # y=alt.Y('y:Q', axis=alt.Axis(title='Tweede factor')),
        x=alt.X('x:Q', axis=None),
        y=alt.Y('y:Q', axis=None),
        color=alt.Color("partij", scale = alt.Scale(domain=parties,range= [party_colors[p] for p in parties]), legend=None)
        )

        text = points.mark_text(
            align='left',
            baseline='middle',
            size=18,
            dx=3,
            dy=0
            # opacity=0.5
        ).encode(
            text='partij:N'
        ).transform_calculate(x='datum.x+ random()*0.1',y='datum.y+ (random()-0.5)*0.3')

        chart = (points + text).configure_view(
            strokeWidth=0)
        chart.configure_axis(
            labelFontSize=14,
            titleFontSize=14,
            grid=False).configure_title(fontSize=66)

        st.write(f'Stemgedrag op onderwerp {selected_topic}, {num_moties} moties, grafiek {round(sum(explained_variance_ratio_)*100)}% betrouwbaar')

        with st.beta_expander("⚙️ - Uitleg ", expanded=False):
            st.write(
                """
            Deze techniek heet Pricipal Component Analysis en probeert variatie op veel dimensies (in dit geval veel moties)
            terug te brengen naar minder dimensies (in dit geval twee, een x en een y as). Als je bijvoorbeeld twee partijen hebt die
            altijd precies tegenovergesteld stemmen dan heb hoef je niet heel veel verschillende moties te visualiseren, maar kan je gewoon de twee tegenover elkaar
            op één as tekenen.

            De afstand tussen twee partijen geeft aan hoe verschillend ze stemmen. 
            Het betrouwbaarheid percentage geeft aan hoeveel van de variatie in het stemgedrag wordt verklaard door de grafiek. Hoe lager dit is des te minder waarde
            je eraan moet hechten. 
            
            Een voorbeeld: stel dat twee partijen precies op hetzelfde punt staan, dan betekent dit bij een betrouwbaarheid van 100% dat ze identiek stemmen.
            Maar als het percentage 50% is betekent dat er nog steeds flink wat variatie is in het stemgedrag is dat niet wordt verklaard door de grafiek.
                """
            )
        return chart

def aantal_moties_chart(df):
    # Overview of topic distribution over all years
    return alt.Chart(df).mark_bar().encode(
        x=alt.X('Indienende_partij:O', sort='-y',title=None),
        y=alt.Y('Aantal moties:Q',title=None),
        color=alt.Color('BesluitTekst:N',
                scale=alt.Scale(
                domain=['Aangenomen','Verworpen'],
                range=['green', 'red']),
                legend=alt.Legend(orient="top",title=None, labelFontSize=14))
        # sort=alt.EncodingSortField('Aantal moties', order='descending'))
        # order=alt.Order('Aantal moties:Q',sort='descending')
    ).configure_axis(
            labelFontSize=14,
            titleFontSize=14,
            grid=False).configure_view(
            strokeWidth=0)

model = load_model()
df = load_df()


search_term = st.text_input('Kies je zoekterm(en)', '')

# select relevant topic topic
if search_term != '':
    try:
        topic_words, word_scores, topic_scores, topic_nums = model.search_topics(keywords= search_term.split() , num_topics=3)
        error = False
    except:
        st.write('(Een van de) woorden komt niet voor in de ingediende moties. Probeer opnieuw')
        error = True

    if not error:
        st.markdown(f'## Onderwerpen die het beste passen bij {search_term}')
        with st.beta_expander("⚙️ - Uitleg ", expanded=False):
            st.write(
                """
            Het Top2Vec algoritme heeft moties (op basis van de woorden) automatisch geclustert in bijna 250 onderwerpen.
            Per onderwerp worden de woorden weergegeven die het meest onderscheidend zijn. Lees de woorden door dan krijg je een idee wat er met het onderwerp ongeveer bedoeld wordt.
            
            Je kan klikken op de andere onderwerpen om hier de resultaten van te zien.
            Je kan ook verder filteren met het linkermenu (pijltje linksboven voor mobiele gebruikers).
                """
            )
        selected_topic = topic_nums[0]
        selected_topic_summary = ' '.join(word for word in topic_words[0][:3])
        for i, (topic, topic_num) in enumerate(zip(topic_words, topic_nums)):
            # st.write('Onderwerp ', topic_nums[i], ' Match: ', round(topic_scores[i]*100), '\n\n ',' '.join(word for word in topic[:20]))
            # st.write('Onderwerp ', topic_nums[i], ' Match: ', round(topic_scores[i]*100))
            if st.button(' '.join(word for word in topic[:20]), key=i):
                selected_topic = topic_num


        # cted_soort = st.radio("Wat voor soort moties: ", (['opwarming aarde broeikasgassen milieuraad co_reductie co uitstoot co_uitstoot klimaatakkoord ets emissies klimaat klimaatdoelen kabinetsaanpak_klimaatbeleid reductie parijs klimaatbeleid wereldwijde duurzame_ontwikkeling doelstelling','opwarming aarde broeikasgassen milieuraad co_reductie co uitstoot co_uitstoot klimaatakkoord ets emissies klimaat klimaatdoelen kabinetsaanpak_klimaatbeleid reductie parijs klimaatbeleid wereldwijde duurzame_ontwikkeling doelstelling']), key=6)
        

        st.sidebar.markdown('Gebruik deze filters om verder te filteren. De grafieken en moties updaten vanzelf')
        selected_soort = st.sidebar.radio("Motie uitkomst: ", (['Aangenomen en verworpen','Aangenomen', 'Verworpen']), key=3)
        selected_party = st.sidebar.radio("Indienende partij: ", (['Alle partijen'] + sorted(parties)), key=2)
        selected_year= st.sidebar.radio("Ingediend in: ", (['Alle jaren'] + ['2017', '2018', '2019', '2020']), key=3)

        max_moties = st.sidebar.slider('maximaal aantal weergegeven moties', 0, 20,5)

        # select data and plot charts
        source = get_df_slice(df)

        st.markdown(f'## "{selected_topic_summary}"')
        st.write(len(source), 'moties ingediend')
        chart = aantal_moties_chart(source.groupby(['Indienende_partij', 'BesluitTekst']).size().reset_index(name='Aantal moties'))
        st.altair_chart(chart, use_container_width=True)

        # width and height does not work altair/streamlit
        if len(source)>2:
            st.markdown(f'## Stemgedrag van partijen op deze {len(source)} moties')
            st.altair_chart(pca_topic(source, selected_topic, 'Rutte III', twodim=True), use_container_width=True  )
        if len(source)>0:
            st.markdown(f'## Moties die het beste passen bij dit onderwerp')

            topic_moties = list(source[(source['Topic_initial']==selected_topic)].index)
            topic_scores = list(source[(source['Topic_initial']==selected_topic)]['Topic_score'])
            for i in range(min(max_moties, len(source))):
                motie_id = topic_moties[i]
                summary = f"Ingediend door {df.loc[motie_id,'Indienende_persoon_partij']}"
                result = f"Resultaat: {df.loc[motie_id,'BesluitTekst']}"
                voor = f"Voor: {', '.join(df.loc[motie_id,'Partijen_Voor'])}"
                tegen = f"Tegen: {', '.join(df.loc[motie_id,'Partijen_Tegen'])}"
                st.write(summary, '  \n', result, '  \n', voor, '  \n', tegen)
                st.text_area('Inhoud van de motie:', df.loc[motie_id,'Text'], height=500, key=i)
        else:
            st.markdown(f'### Geen moties gevonden (staan er filters aan?)')


    with st.beta_expander("⚙️ - Thanks & feedback ", expanded=False):
        st.markdown(
                """
         
            Heb je een inzichten opgedaan, feedback op de app of wil je contact opnemen? Laat het weten, bijvoorbeeld door te reageren op m'n [LinkedIn](https://www.linkedin.com/in/jessevanelteren/) post.

            Dank gaat uit naar:
        
            * :sun_with_face: [Tweede Kamer Open Data Portaal](https://opendata.tweedekamer.nl/) voor het beschikbaar maken van de moties via een API
            * :sun_with_face: [Willem Glasbergen](https://www.linkedin.com/in/willemglasbergen) die me via [Longhow Lam](https://www.linkedin.com/posts/longhowlam_top2vec-stem-helper-activity-6772061735844098048-zKd6) op Top2Vec attendeerde
            * :sun_with_face: Dimo Angelov, bedenker en ontwikkelaar van [Top2Vec](https://github.com/ddangelov/Top2Vec)
            * :sun_with_face: [Streamlit](https://streamlit.io/) voor het maken van zo'n geweldige library

            [![License: Creative Commons Naamsvermelding-GelijkDelen 4.0 Internationaal-licentie](https://i.creativecommons.org/l/by-sa/4.0/80x15.png)](https://creativecommons.org/licenses/by-sa/3.0/) 2021 Jesse van Elteren
                """
        )




# %%

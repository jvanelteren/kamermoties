# To add a new cell, type '#%%'
# To add a new markdown cell, type '#%% [markdown]'
#%%
from IPython import get_ipython


#%%
import urllib.request
import json
import numpy as np
import pandas as pd
import altair
import pprint
import os, subprocess, sys,codecs, locale
import re
import traceback
#get_ipython().run_line_magic('matplotlib', 'inline')


#%%
pd.options.display.width = 250
pd.options.display.max_colwidth = 250
pd.options.display.max_rows = 250

FIRST_ONLY = True
INCLUDE_PDF = True
START = 0
MAXIMUM = 100000
path = 'pdf/' # where to store pdfs of motions


#%%
# some helper functions to obtain the right output from API
def remove_spaces(input):
    return input.replace(' ', '%20')

def pdf_to_text(pdf_path):
    #sys.setdefaultencoding("utf-8")
    os_encoding = locale.getpreferredencoding()
    args = ["pdftotext.exe",pdf_path, "-"]
    res = subprocess.run(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    output = (res.stdout).decode(os_encoding,'ignore')
    return(output)

def retrieve_and_save_pdf(id):
    if not os.path.exists(path+id+'.pdf'):
        base = 'https://gegevensmagazijn.tweedekamer.nl/OData/v4/2.0/'
        url = 'Document('+id+')/Resource'
        end = ''
        response = urllib.request.urlopen(remove_spaces(base+url))
        data = response.read()      # a `bytes` object
        f = open(path+id+'.pdf', 'w+b')
        #binary_format = bytearray(byte_arr)
        f.write(data)
        f.close()
    else:
        pass
        #print('file already downloaded')

def query_API(skip):
    base = 'https://gegevensmagazijn.tweedekamer.nl/OData/v4/2.0/'
    url = "/Zaak?$filter= Soort eq 'Motie'&$expand=Besluit($expand=Stemming),Document,ZaakActor,Agendapunt&$count=true&$skip="+skip
    end = '&$format=application/json;odata.metadata=full'
    response = urllib.request.urlopen(remove_spaces(base+url))
    data = response.read()      # a `bytes` object
    return(json.loads(data))
#retrieve_and_save_pdf('1b7f8f8c-4579-4103-9f51-9280358f7b8a')
#a = pdf_to_text(path+'3c2dbc59-beb1-42d8-9be4-e09238f20a97.pdf')
#print(a)


#%%
def add_API_to_dict(data):
    
    try:
        for z in data['value']:
            #print('\n\n','nummer',z['Nummer'])
            info[z['Nummer']]={'Titel':z['Titel'],
                              #'Status':z['Status'],
                              'Onderwerp':z['Onderwerp'],
                              'Vergaderjaar':z['Vergaderjaar']}

            for b in z['Besluit']:
                info[z['Nummer']]['BesluitTekst']=b['BesluitTekst']
                info[z['Nummer']]['StemmingsSoort']=b['StemmingsSoort']
                info[z['Nummer']]['BesluitSoort']=b['BesluitSoort']

                #print('\n','besluit',b['BesluitTekst'])
                if 'aangenomen' in b['BesluitSoort'] or 'verworpen' in b['BesluitSoort']:        
                    for s in b['Stemming']:
                        #print(s)
                        info[z['Nummer']]['Stem_'+s['ActorFractie']]=(s['Soort'])
                        info[z['Nummer']]['Aantal_stemmen_'+str(s['ActorFractie'])]=(s['FractieGrootte'])
                        #info[z['Nummer']]['Vergissing_'+str(s['ActorFractie'])]=(s['Vergissing']) niet nodig, want Stem is altijd de gecorrigeerde stem (dus niet de vergissing)
                    #print('stemmingenverwerkt')
                    break
            for a in z['Agendapunt']:
                info[z['Nummer']]['AgendapuntOnderwerp']=a['Onderwerp']

            for d in z['Document']:
                test[z['Nummer']] = test[z['Nummer']].append(d['Soort']) 
                if d['Soort']=='Motie':
                    #print('\n','doc',d)
                    info[z['Nummer']]['doc_Id']=d['Id']
                    info[z['Nummer']]['Volgnummer']=d['Volgnummer']
                    info[z['Nummer']]['Datum']=d['Datum']
                    if INCLUDE_PDF:
                        #print('doing pdf work')
                        retrieve_and_save_pdf(d['Id'])
                        info[z['Nummer']]['Text']=pdf_to_text(path+d['Id']+'.pdf')

            for za in z['ZaakActor']:
                #print('\n','za',za)
                if za['Relatie'] == 'Indiener' and za['ActorFractie']:
                    #info[z['Nummer']]['Indiener_persoon_'+za['ActorFractie']]=za['ActorNaam']
                    info[z['Nummer']]['Indiener_'+za['ActorFractie']]=1
                
                if za['Relatie'] == 'Medeindiener' and za['ActorFractie']:
                    #info[z['Nummer']]['Medeindiener_persoon_'+za['ActorFractie']]=za['ActorNaam']
                    info[z['Nummer']]['Medeindiener_'+za['ActorFractie']]=1
    except Exception as e: 
        print(e)
        print(z['Nummer'])
        pprint.pprint(z) 
        sys.exit()


#%%
# main routine to query api
from collections import defaultdict
test = defaultdict(list)
info = {}
print ('started')
data = query_API(str(START))
count = data['@odata.count']
add_API_to_dict(data)
skip = START +  250

if not FIRST_ONLY:
    while skip < count and skip < MAXIMUM:
        print('query for ',skip)
        data = query_API(str(skip))
        add_API_to_dict(data)
        skip +=250
print('finished')


#%%
import pickle
with open('moties_unprocessed.pickle', 'wb') as handle:
    pickle.dump(info, handle, protocol=pickle.HIGHEST_PROTOCOL)

#%% [markdown]
# ### Make dataframe
# 

#%%
import pickle
file = open("moties_unprocessed.pickle","rb")
info = pickle.load(file)
print(len(info))

#%% [markdown]
# ### Preprocessing

#%%
df = pd.DataFrame(info).T

# sort columns
column_list = df.columns.values # current columns
column_order = ['Titel','AgendapuntOnderwerp','Onderwerp','Vergaderjaar','Datum','StemmingsSoort','BesluitSoort','BesluitTekst','doc_Id','Volgnummer'] # desired columns
if INCLUDE_PDF: column_order.append('Text')
for c in column_list: # don't throw away columns
    if c not in column_order: column_order.append(c)
df = df[column_order]


#%%
df['Text'] = df['Text'].astype(str)
df['Text'] = df['Text'].str.replace('\xad', '')
df['Text'] = df['Text'].str.replace('\n', '')
df['Text'] = df['Text'].str.replace('\r', '')
df['Text'] = df['Text'].str.replace('-', '')


#%%
df['doc_Id'].isnull().sum()


#%%
df.loc['2018Z02894']


#%%
no_text = []
error_processing = []

def preprocessing_motie(text,doc_id,i): # remove start and end of document to only include relevant text

    try:
        #print('row',i," doc_id",doc_id)
        #print(df.loc[i,'Text'])
        if text and doc_id:
            #regex = re.findall(r"\d+ \d+.*?\r\n\r\n(.*?)\r.*?VAN.*gehoord de beraadslaging,(.*)gaat over tot de orde van",text,re.DOTALL)
            #regex = re.findall(r"\d{4} \d{4}.*?\r\n\r\n(.*?)\r.*?VAN.*gehoord de beraadslaging,(.*)",text,re.DOTALL)
            #regex = re.findall(r"\d{8}.+?\r\r(.*?)\r\r.+?gehoord de beraadslaging(.+)\ren gaat",text,re.DOTALL)
            regex = re.findall(r"\d{4}(.*?)Nr.*gehoord de beraadslaging(.*)",text,re.DOTALL)

            #print('\n',regex)
            regex = (' '.join(regex[0]))
            return regex
        else:
            print ('bla',i)
            no_text.append(i)
            return ""
    except Exception:
        print('error')
        print(traceback.print_exc())
        print(i)
        error_processing.append(i)
        return ""


case = '2018Z02894'
result = preprocessing_motie(df.loc[case]['Text'],case,case)
df.loc[case]['Text'],'\n result',result


#%%
df['Text_processed'] = np.vectorize(preprocessing_motie)(df['Text'],df['doc_Id'],df.index)


#%%
len(no_text),len(error_processing)


#%%
#verwijder hoofdelijke stemmingen
print (len(df))
df.drop(df[df['StemmingsSoort'] == 'Hoofdelijk'].index, inplace=True)
print (len(df))

#verwijder moties zonder stemming
print (len(df))
print(df['BesluitSoort'].value_counts())
df = df[((df['BesluitSoort'] == 'Stemmen - aangenomen') | (df['BesluitSoort'] == 'Stemmen - verworpen'))]
print (len(df))

#recode besluitsoort naar -1 en 1
print(df['BesluitSoort'].value_counts())
df['BesluitSoort']=df['BesluitSoort'].replace({'Stemmen - verworpen':'0','Stemmen - aangenomen':'1'})
df['BesluitSoort']=pd.to_numeric(df['BesluitSoort'])
print(df['BesluitSoort'].value_counts())

#recode voor en tegen naar 1 en -1
stem_column = [c for c in column_list if 'Stem_' in c]
print(df['Stem_50PLUS'].value_counts())
df[stem_column]=df[stem_column].replace({'Tegen':'-1','Voor':'1','Niet deelgenomen':np.nan})
print(df['Stem_50PLUS'].value_counts())

#cast to datetime and sort old to new
df['Datum'] = pd.to_datetime(df['Datum'])
df.sort_values('Datum',inplace=True)


#%%
#bereken voor en tegenstemmen
aantal_stemmen_column = [c for c in column_list if 'Aantal' in c]

for i in range(len(stem_column)):
    df[stem_column[i]]=pd.to_numeric(df[stem_column[i]])
    df[aantal_stemmen_column[i]]=pd.to_numeric(df[aantal_stemmen_column[i]])

res = np.multiply(df[stem_column],df[aantal_stemmen_column])
voor = res[res > 0].sum(axis=1)
tegen = abs(res[res < 0].sum(axis=1))
df['Voor'], df['Tegen'] = voor,tegen
df['Delta'] = abs(df['Voor']- df['Tegen'])
df['Sum'] = (df['Voor']+ df['Tegen'])


#%%
def get_largest_parties(year=2018,top=False):
    tmp = df[(df['Vergaderjaar']==str(year)+'-'+str(year+1))]
    parties = [p for p in tmp.columns if 'Aantal_stemmen' in p][1:]
    tmp = tmp[parties].mean().sort_values(ascending=False)
    tmp = tmp[tmp.notna()]
    tmp.index = tmp.index.str[15:]
    if top: return tmp[:top].index
    else: return tmp.index
get_largest_parties()


#%%
indiener_column = [c for c in column_list if 'Indiener' in c]
medeindiener_column = [c for c in column_list if 'Medeindiener' in c]


#%%
#df = df[df['Onderwerp'].str.contains('klimaat')]
import pickle
with open('moties_processed_df.pickle', 'wb') as handle:
    pickle.dump(df, handle, protocol=pickle.HIGHEST_PROTOCOL)

#%% [markdown]
# #### Exploratory analysis

#%%
import altair as alt
import numpy as np
import pandas as pd
import sklearn
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
import altair as alt
import pandas as pd


#%%
import pickle
file = open("moties_processed_df.pickle","rb")
df = pickle.load(file)
print(len(df))

#%% [markdown]
# ### Welke partijen vertonen gelijkwaardig stemgedrag?

#%%
year = 2010
for start_year in range(year,2019):

    source_year = df.loc[df['Vergaderjaar'] == str(start_year)+'-'+str(start_year+1)][stem_column].dropna(axis=1, how='all').T
    X_year = SimpleImputer(strategy='constant', fill_value=0).fit_transform(source_year)

    pca = PCA(n_components=2)
    pca = pca.fit(X_year)
    print('PCA results for',start_year,len(source_year))

    print(pca.explained_variance_ratio_,pca.explained_variance_ratio_.sum())  
    #print(pca.singular_values_)  

res_year = pca.transform(X_year)

source = pd.DataFrame(res_year)
source['label'] = source_year.T.columns
source = source.rename(index=str, columns={0: "x", 1: "y"})                      

points = alt.Chart(source).mark_point().encode(
    x='x:Q',
    y='y:Q',
    tooltip=['label']
)
text = points.mark_text(
    align='left',
    baseline='middle',
    dx=np.random.uniform(0,10),
    dy=np.random.uniform(0,10),
    opacity=0.5
).encode(
    text='label'
    
)
points + text


#%%
year = 2018
for start_year in range(year,2019):

    source_year = df.loc[df['Vergaderjaar'] == str(start_year)+'-'+str(start_year+1)][stem_column].dropna(axis=1, how='all').T
    X_year = SimpleImputer(strategy='constant', fill_value=0).fit_transform(source_year)

    pca = PCA(n_components=1)
    pca = pca.fit(X_year)
    print('PCA results for',start_year,len(source_year))

    print(pca.explained_variance_ratio_,pca.explained_variance_ratio_.sum())  
    #print(pca.singular_values_)  

res_year = pca.transform(X_year)

source = pd.DataFrame(res_year)
source['label'] = source_year.T.columns
source = source.rename(index=str, columns={0: "x"}).sort_values('x',ascending=False)
order = source['label']
order = order.str[5:]
alt.Chart(source).mark_bar().encode(
    x=alt.X('label:N',sort=alt.EncodingSortField(field="x", op="count", order='ascending')),
    y='x:Q'
)

#%% [markdown]
# ### Hoe spannend zijn de stemmingen? (delta van 0 is evenveel voor-als tegenstemmers)
#%% [markdown]
# ### Stemmen we voor of tegen?

#%%
source = pd.DataFrame({'Voor':df.groupby(['Vergaderjaar'])['Voor'].mean().values,
          'Tegen':df.groupby(['Vergaderjaar'])['Tegen'].mean().values}).reset_index()


#%%
source = source.melt('index',var_name='Stem', value_name='Gemiddelde')


#%%
alt.Chart(source).mark_line().encode(
    x='index:N',
    y='Gemiddelde:Q',
    color='Stem:N'
)

#%% [markdown]
# ### Van Koorten en de Bie: Welke partij is de 'tegenpartij'?

#%%
startjaar = 2018
source = df
source = df.replace(-1,0)

source = source.groupby(['Vergaderjaar']).mean()[['Stem_'+p for p in get_largest_parties(startjaar,15)]]
#res =  res.sub(res.mean(axis=1).values,axis=0)
source.T.sort_values(str(startjaar)+'-'+str(startjaar+1))
source = source.reset_index()


#%%
source = source.melt('Vergaderjaar',var_name='Stem', value_name='Stemgedrag')
source.loc[source['Vergaderjaar']=='2018-2019']


#%%
alt.Chart(source,width=600).mark_line().encode(
    x='Vergaderjaar:N',
    y='Stemgedrag:Q',
    color='Stem:N',
)

#%% [markdown]
# ### Welke partijen dienen het meeste moties in en hoe succesvol zijn ze?

#%%
def get_summary_stats(year,df):
    df = df.dropna(axis='columns',how='all')
    indiener_column = [c for c in df.columns if 'Indiener' in c]
    indiener = {p:p[9:] for p in indiener_column}
    df = df.rename(indiener,axis=1)
    largest_parties = get_largest_parties(year,15)
    indiener_column = [i for i in indiener.values() if i in largest_parties]
    aantal_stemmen_column = [c for c in df.columns if 'Aantal_stemmen' in c]
    
    
    success_rate = np.multiply(df[indiener_column],df['BesluitSoort'][:, np.newaxis]).mean(axis=0).sort_index()
    success_rate = success_rate[success_rate.notna()]
    #print('success_rate',len(success_rate),'\n',success_rate.sort_values(),'\n')

    aantal_moties = df[indiener_column].sum().sort_index()
    aantal_moties = aantal_moties[aantal_moties.notna()]
    #print('aantal moties', len(aantal_moties),'\n',aantal_moties,'\n')

    aantal_succesvolle_moties = np.multiply(success_rate,aantal_moties)
    aantal_succesvolle_moties = aantal_succesvolle_moties[aantal_succesvolle_moties.notna()]
    #print('number_successful',aantal_succesvolle_moties,'\n')
    
    aantal_niet_succesvolle_moties = np.multiply((1-success_rate),aantal_moties)
    aantal_niet_succesvolle_moties = aantal_niet_succesvolle_moties[aantal_niet_succesvolle_moties.notna()]

    aantal_zetels = df[aantal_stemmen_column].mean()
    aantal_zetels = aantal_zetels[['Aantal_stemmen_'+p for p in aantal_moties.index]]
    aantal_zetels.index = aantal_moties.index
    aantal_zetels = aantal_zetels[aantal_zetels.notna()]
    #print('aantal zetels',len(aantal_zetels),'\n',aantal_zetels)

    aantal_zetels.index = aantal_moties.index
    aantal_moties_zetel = np.divide(aantal_moties,aantal_zetels)
    #print(aantal_moties_zetel.sort_values())

    aantal_succesvolle_moties_zetel = np.divide(aantal_succesvolle_moties,aantal_zetels)
    #print(aantal_succesvolle_moties_zetel.sort_values())
    aantal_niet_succesvolle_moties_zetel = np.divide((aantal_moties-aantal_succesvolle_moties),aantal_zetels)
    
    return success_rate,aantal_moties,aantal_succesvolle_moties,aantal_niet_succesvolle_moties, aantal_zetels, aantal_moties_zetel,aantal_succesvolle_moties_zetel,aantal_niet_succesvolle_moties_zetel

year = 2010
party_perf = pd.DataFrame({})
res_columns = ['success_rate','aantal_moties','aantal_succesvolle_moties','aantal_niet_succesvolle_moties', 'aantal_zetels', 'aantal_moties_zetel','aantal_succesvolle_moties_zetel','aantal_niet_succesvolle_moties_zetel']
for start_year in range(year,2019):
    result = get_summary_stats(start_year,df.loc[df['Vergaderjaar'] == str(start_year)+'-'+str(start_year+1)])
    for i,res in enumerate(result):
        df_temp = pd.DataFrame(res).reset_index()
        df_temp['stat']=res_columns[i]
        df_temp['year']=str(start_year)+'-'+str(start_year+1)
        #print(df_temp)
        #print(df_temp)
        party_perf = party_perf.append(df_temp, ignore_index = True)
party_perf = party_perf.rename({'index':'partij',0:'value'},axis=1)        


#%%
# Step 1: create the line
line = alt.Chart(party_perf).mark_line(interpolate="basis").encode(
    x='year:N',
    y='value:Q',
    color='partij:N'
).transform_filter(alt.FieldEqualPredicate(field='stat', equal='aantal_succesvolle_moties'))


# Step 2: Selection that chooses nearest point based on value on x-axis
nearest = alt.selection(type='single', nearest=True, on='mouseover',
                            fields=['year'])


# Step 3: Transparent selectors across the chart. This is what tells us
# the x-value of the cursor
selectors = alt.Chart().mark_point().encode(
    x="year:N",
    opacity=alt.value(0),
).add_selection(
    nearest
)

# Step 4: Add text, show values in Sex column when it's the nearest point to 
# mouseover, else show blank
text = line.mark_text(align='left', dx=3, dy=-3).encode(
    text=alt.condition(nearest, 'partij:N', alt.value(' '))
)

# Layer them all together
chart = alt.layer(line, selectors, text, data=party_perf, width=300)

chart


#%%
alt.Chart(party_perf).mark_bar().encode(
    column='year',
    x='value:Q',
    y='partij',
    color='stat'
).properties(width=60).transform_filter(alt.FieldOneOfPredicate(field='stat', oneOf=['aantal_succesvolle_moties','aantal_niet_succesvolle_moties']))


#%%


alt.Chart(party_perf).mark_bar().encode(
    column='year',
    x='value:Q',
    y='partij',
    color='stat'
).properties(width=60).transform_filter(alt.FieldOneOfPredicate(field='stat', oneOf=['aantal_succesvolle_moties_zetel','aantal_niet_succesvolle_moties_zetel']))


#%%
year = 2010
medeindiener_perc = pd.DataFrame({})
for start_year in range(year,2019):
    moties_in_year = df.loc[(df['Vergaderjaar'] == str(start_year)+'-'+str(start_year+1))]
    moties_in_year = moties_in_year.dropna(axis=1, how='all')
    indieners = [c for c in moties_in_year.columns if 'Indiener' in c]
    medeindieners = [c for c in moties_in_year.columns if 'Medeindiener' in c]
    stem = ['Stem_'+c[13:] for c in medeindieners]
    for p in indieners:
        partij_moties =  moties_in_year.loc[(moties_in_year[p]==1)]
        #print(partij_moties[medeindieners].sum(),partij_moties[stem].notna().sum(), '/n')
        res = (partij_moties[medeindieners].sum()/partij_moties[stem].notna().sum().values)
        #print(res)
        res = pd.DataFrame({'support':res.index.str[13:],'percentage':res.values})
        res['partij']= p[9:]
        res['year']=str(start_year)+'-'+str(start_year+1)
        #print(res)
        medeindiener_perc = medeindiener_perc.append(res)
        


#%%
medeindiener_perc = medeindiener_perc[medeindiener_perc['support'].isin(order)]
medeindiener_perc = medeindiener_perc[medeindiener_perc['partij'].isin(order)]

alt.Chart(medeindiener_perc).mark_rect().encode(
    column='year:N',
    x=alt.X('support:N', sort=list(order)),
    y=alt.Y('partij:N', sort=list(order)),    
    color=alt.Color('percentage', scale=alt.Scale(scheme='greens'))
)#.properties(width=200)


#%%
year = 2010
stem_perc = pd.DataFrame({})
for start_year in range(year,2019):
    moties_in_year = df.loc[(df['Vergaderjaar'] == str(start_year)+'-'+str(start_year+1))]
    moties_in_year = moties_in_year.dropna(axis=1, how='all')
    indieners = [c for c in moties_in_year.columns if 'Indiener' in c]
    stem = [c for c in moties_in_year.columns if 'Stem' in c][1:]
    
    #print(stem)
    for p in indieners:
        #print(p,p[9:])
        partij_moties =  moties_in_year.loc[(moties_in_year[p]==1)] 
        
        partij_moties = partij_moties.replace(-1,0)
        #print(partij_moties['Stem_FvD'])
        #print('bla',partij_moties[stem].sum(),len(partij_moties))
        res = (partij_moties[stem].sum()/partij_moties[stem].notna().sum())
        
        #print('res',res)
        res = pd.DataFrame({'support':res.index.str[5:],'percentage':res.values})
        res['partij']= p[9:]
        res['year']=str(start_year)+'-'+str(start_year+1)
        stem_perc = stem_perc.append(res)


#%%
stem_perc = stem_perc[stem_perc['support'].isin(order)]
stem_perc = stem_perc[stem_perc['partij'].isin(order)]

alt.Chart(stem_perc).mark_rect().encode(
    column = 'year',
    x=alt.X('support:N', sort=list(order)),
    y=alt.Y('partij:N', sort=list(order)),
    color=alt.Color('percentage', scale=alt.Scale(scheme='redyellowgreen'))
)#.properties(width=200)

#%% [markdown]
# ### Voorspel stemuitslag

#%%
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
start_year = 2008

scores = []
for s in range(start_year,2019):
    X = df.loc[df['Vergaderjaar'] == str(s)+'-'+str(s+1)][indiener_column+medeindiener_column].dropna(how='all',axis=1)
    X_imp = SimpleImputer(strategy='constant', fill_value=0).fit_transform(X)
    y = df.loc[df['Vergaderjaar'] == str(s)+'-'+str(s+1)]['BesluitSoort']
    X_train, X_test, y_train, y_test = train_test_split(X_imp, y, test_size=0.5, shuffle=False)


    clf = LogisticRegression(random_state=0, solver='lbfgs',
                             multi_class='ovr').fit(X_train, y_train)

    print(clf.score(X_test, y_test),y.value_counts()[0]/len(y))
    scores.append(clf.score(X_test, y_test))
    print (confusion_matrix(y_test, clf.predict(X_test), sample_weight=None))
print (np.mean(scores))


#%%



#%%



#%%



#%%



#%%



#%%
test


#%%




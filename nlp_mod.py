import re
import nltk
from nltk.stem import WordNetLemmatizer
from sentence_transformers import SentenceTransformer
from scipy import spatial
import numpy as np

nltk.download('stopwords', quiet = True)
en_stop = set(nltk.corpus.stopwords.words('english'))
stemmer = WordNetLemmatizer()
BERTModel = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
KNN_K = 3
SIMILARITY_THRESHOLD = 0.65
STEP_SIMILARITY_THRESHOLD = 0.65

stopwords_to_exclude = ['to','not','no']
stopwords_to_add = ['error']
for word in stopwords_to_exclude: en_stop.remove(word)
for word in stopwords_to_add: en_stop.add(word)  
    
def preprocess_text(str_):
    '''
        preprocess_text(str_)
        preprocessing for one-sentence documents.
        >>> parameters:
            - str_ | string | one line document
        >>> returns:
            - *transformed string
    '''
    str_ = re.sub(r'\W', ' ', str(str_))         # Remove all the special characters
    str_ = re.sub(r'\s+[a-zA-Z]\s+', ' ', str_)  # remove all single characters
    str_ = re.sub(r'\^[a-zA-Z]\s+', ' ', str_)   # Remove single characters from the start
    str_ = re.sub(r'\s+', ' ', str_, flags=re.I) # Substituting multiple spaces with single space
    str_ = str_.lower()                          # Converting to Lowercase

    tokens = str_.split()    
    tokens = [word for word in tokens if word not in en_stop]
    tokens = [stemmer.lemmatize(word) for word in tokens]
    
    return ' '.join(tokens)

def extract_keywords(str_):
    '''
        extracts relevant words from string for 
    '''
    doc = nlp(str_)
    str_ = []
    for token in doc:
        if token.pos_ in ['ADJ','ADP','AUX','VERB','PART']:#,'NOUN'
            str_.append(token.text)
    return ' '.join(str_)
    
def assign_incident_subclass(short_desc, logs, APP_NAME):
    '''
        assign_incident_subclass(short_desc, logs, APP_NAME)
        assigns the new incident to a named cluster. clusters are sequentially created for each app based on text similarity.
        >>> parameters:
            - short_desc | string | short description entry of new incident
            - logs | dataframe | ...of incident logs of this app
        >>> returns
            - subclass | string | name of assigned class
    '''
    subclass = 'NO_MATCH'
    if logs.shape[0] == 0:
        # check for cold start
        subclass = APP_NAME + '_SubClass_1'
    else:
        if error_code_available(short_desc) != 'NO_MATCH':
            subclass = match_by_error_code(short_desc,logs)
        if subclass == 'NO_MATCH':
            subclass = match_by_text_similarity(short_desc,logs)
        if subclass == 'NO_MATCH':
            subclass = APP_NAME + '_SubClass_' + str(logs['SubClass'].unique().size + 1)
    
    return subclass

def error_code_available(str_):
    '''
        error_code_available(str_)
        finds any error code mentions in the short description string.
        >>> parameters:
            - str_ | string | one line document
        >>> returns:
            - res_ | string | search result
    '''
    search_res = re.findall('error *\d+',str_)
    
    if len(search_res) > 0:
        return search_res[0]
    else:
        return 'NO_MATCH'
    
def match_by_error_code(short_desc, logs):
    '''
        match_by_error_code(short_desc, logs)
        identifies a row match if the same error code is present
        >>> parameters:
            - short_desc | string | short description entry of new incident
            - logs | dataframe | log of incidents in this app so far
        >>> returns:
            - subclass | string | SubClass if found else NO_MATCH
    '''
    subclass = 'NO_MATCH'
    error = error_code_available(short_desc)
    for i in range(logs.shape[0]):
        if error in logs['Short Description'].iloc[i]:
            subclass = logs['SubClass'].iloc[i]
            break
    return subclass

def match_by_text_similarity(short_desc, logs):
    '''
        match_by_text_similarity(short_desc, logs)
        finds the cosine similarities of the description of the new incident to the ones in records.
        if insufficient similarity, returns NO_MATCH 
        else identify class by KNN
        >>> parameters:
            - short_desc | string | short description entry of new incident
            - logs | dataframe | log of incidents in this app so far
        >>> returns:
            - subclass | string | SubClass if found else NO_MATCH
    '''
    subclass = 'NO_MATCH'
    short_desc = BERTModel.encode(short_desc)
    logs['similarity_score'] = logs['Short Description'].apply(lambda x: 1 - np.abs(spatial.distance.cosine(BERTModel.encode(x), short_desc)))
    
    clusters = logs[['similarity_score','SubClass']]
    clusters = clusters[clusters['similarity_score'] > SIMILARITY_THRESHOLD]

    if clusters.shape[0] > 0:
        subclass = clusters.sort_values('similarity_score', ascending = False).head(KNN_K)['SubClass'].value_counts().index[0]

    return subclass

def match_by_step_similarity(short_desc, logs):

    subclass = 'NO_MATCH'
    short_desc = BERTModel.encode(short_desc)
    logs['similarity_score'] = logs['workNote'].apply(lambda x: 1 - np.abs(spatial.distance.cosine(BERTModel.encode(x), short_desc)))
    
    clusters = logs[['similarity_score','stepClass']]
    clusters = clusters[clusters['similarity_score'] > STEP_SIMILARITY_THRESHOLD]

    if clusters.shape[0] > 0:
        subclass = clusters.sort_values('similarity_score', ascending = False).head(KNN_K)['stepClass'].value_counts().index[0]

    return subclass

def assign_step_subclass(short_desc, logs):

    subclass = 'NO_MATCH'
    if logs.shape[0] == 0:
        # check for cold start
        subclass = 'SubClass_1'
    else:
        subclass = match_by_step_similarity(short_desc,logs)
        if subclass == 'NO_MATCH':
            subclass = 'SubClass_' + str(logs['stepClass'].unique().size + 1)
    return subclass

def cluster_and_replace(subP):
    subProcess = subP.reset_index(drop=True)
    subProcess['stepClass'] = 'NO_MATCH'

    for i in range(subProcess.shape[0]):
        clss = assign_step_subclass(subProcess['workNote'].iloc[i], subProcess.loc[0:(i-1)])
        #print(clss)
        subProcess['stepClass'].iloc[i] = clss
    
    rep_dict = {}
    replacements = subProcess.loc[subProcess['workNote'].str.len().sort_values().index].drop_duplicates(subset='stepClass')
    for i in range(replacements.shape[0]):
        rep_dict[replacements['stepClass'].iloc[i]] = replacements['workNote'].iloc[i]

    if len(rep_dict.keys()) > 0:
        subProcess['workNote'] = subProcess.apply(lambda s: rep_dict[s['stepClass']],axis=1)
    #print(rep_dict)
    return subProcess.values.tolist()

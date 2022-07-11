from matplotlib.pyplot import show
import pandas as pd  # pip install pandas openpyxl
import plotly.express as px
from requests import head  # pip install plotly-express
import streamlit as st  # pip install streamlit
import json
import pyodbc
from unidecode import unidecode
# Import module for data manipulation
import pandas as pd
# Import module for linear algebra
import numpy as np
# Import module for Fuzzy string matching
from fuzzywuzzy import fuzz, process
# Import module for regex
import re
# Import module for iteration
import itertools
# Import module for function development
from typing import Union, List, Tuple
# Import module for TF-IDF
from sklearn.feature_extraction.text import TfidfVectorizer,TfidfTransformer, CountVectorizer
# Import module for cosine similarity
from sklearn.metrics.pairwise import cosine_similarity
# Import module for KNN
from sklearn.neighbors import NearestNeighbors


# ML Models
import pandas as pd
import numpy as np
import unidecode
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from itertools import chain #Para a função oov

################################## NEO NLP ##################################

class NeoNLP():
    def __init__(self):
        """
        Inicialização da classe NeoNLP
        Atributo:
        text2vec = CountVectorizer
        """
        self.text2vec = CountVectorizer(strip_accents='unicode')
        self.text2tfidf = TfidfTransformer()

        
    def normalize(self, text):
        """
        função normalize
        Transforma texto para minúsculo e sem acento
        Entrada:
        texto
        Retorno:
        Texto no formato minúsculo e sem acento.
        """
        return unidecode.unidecode(text.lower())
    
    def tokenize(self, text):
        """
        Função tokenize
        Divide o texto em palavras utilizando como separador espaço em branco (múltiplos espaços são considerados como um)
        """
        return text.split(' ')
    
    def normalize_tokenize(self, text):
        """
        Função normalize_tokenize
        Aplica na sequencia a função normalize e depois a função tokenize
        """
        return self.tokenize(self.normalize(text))

    #def fit(self, corpus, categories):
    #    self.df_model = pd.DataFrame(data = [corpus, categories], columns = ['corpus','categories'])
    
    def build_model(self,corpus):
        """
        Função build_model
        Cria o vocabulário a partir de um corpus gerando o dicionário de-para
        Armazena o corpus utilizado no atributo corpus
        """
        self.corpus = corpus.copy()
        self.text2vec.fit(self.corpus)

    def fit(self, description, classification):
        #Prepara o dataframe
        self.df_classification = pd.DataFrame({'description':description,'category':classification})
        self.df_group = self.df_classification.groupby("category").agg({'description':lambda x: ' '.join(x)})
        self.A = self.transform(self.df_group.description.values).T
        self.B = self.text2tfidf.fit_transform(self.A.T).T.toarray()        
        #idf = self.A.count_nonzero
        self.cat2index = dict(zip(self.df_group.index, range(len(self.df_group))))
        self.index2cat = dict(zip(range(len(self.df_group)),self.df_group.index))

    def predict(self, text, category = True, method = 'tf'):
        """
        Função predict
        Realiza a predição de uma categoria a partir de uma base de treinamento
        Requisitos:
            Requer a criação do vocabulário (função build_model)
            Requer antes a chamada da função fit, passando as descrições e as classes (função fit)
        Argumentos:
            category: True or False. por padrão True, retorna o nome da categoria, caso False retorna o número da categoria
            para recuperar o nome pode-se chamar a função index2cat para recuperar o nome da categoria
            method: 
                'tf': utiliza diretamente a matriz termo frequencia para recuperar a categoria, realizando o produto A.T@x e encontrando o argmax
                'tf_norm': utiliza o vetor tf normalizado (A/np.linalg.norm(A,axis=0)).T@x e encontrando o argmax
                'cossim': utiliza a similaridade a partir do cálculo do cosseno (A/np.linalg.norm(A,axis=0)).T@(x/np.linalg.norm(x)) e devolve o argmax
                'tfidf': utiliza o modelo tf normalizado com idf suavizado
                'prob':utiliza um modelo probabilístico orientado por linha
        """
        x = self.transform(text).T
        if method == 'tf':
            result = np.argmax(self.A.T@x,axis=0) 
        elif method == 'tf_norm':
            result = np.argmax((self.A/np.linalg.norm(self.A,axis=0)).T@x,axis=0) 
        elif method == 'cossim':
            result = np.argmax((self.A/np.linalg.norm(self.A,axis=0)).T@(x/np.linalg.norm(x)),axis=0)
        elif method == 'tfidf':
            result = np.argmax(self.B.T@x,axis=0)
        elif method == 'prob':
            result = np.argmax(np.exp(np.log((self.A/self.A.sum(axis=1).reshape(-1,1)) + 1/self.A.sum()).T@x),axis=0)
        else:
            raise f"method {method} invalid, try ['tf','tf_norm','cossim']" 
        return list(map(lambda x: self.index2cat.get(x), result)) if category else result
        #pd.DataFrame({'category':neonlp.cat2index.keys(),'score':(neonlp.A.T@x).T[0]}).sort_values('score',ascending=False)

    def oov(self, raw_documents):
        analyzer = self.text2vec.build_analyzer()
        analyzed_documents = [analyzer(doc) for doc in (raw_documents if type(raw_documents)!=type('') else [raw_documents]) ]
        new_tokens = set(chain.from_iterable(analyzed_documents))
        oov_tokens = new_tokens.difference(set(self.text2vec.vocabulary_.keys()))
        return oov_tokens

    def predict2df(self, text, category = True, method = 'tf'):
        """
        Função predict
        Realiza a predição de uma categoria a partir de uma base de treinamento
        Requisitos:
            Requer a criação do vocabulário (função build_model)
            Requer antes a chamada da função fit, passando as descrições e as classes (função fit)
        Argumentos:
            category: True or False. por padrão True, retorna o nome da categoria, caso False retorna o número da categoria
            para recuperar o nome pode-se chamar a função index2cat para recuperar o nome da categoria
            method: 
                'tf': utiliza diretamente a matriz termo frequencia para recuperar a categoria, realizando o produto A.T@x e encontrando o argmax
                'tf_norm': utiliza o vetor tf normalizado (A/np.linalg.norm(A,axis=0)).T@x e encontrando o argmax
                'cossim': utiliza a similaridade a partir do cálculo do cosseno (A/np.linalg.norm(A,axis=0)).T@(x/np.linalg.norm(x)) e devolve o argmax
        """
        x = self.transform(text).T
        if method == 'tf':
            result = self.A.T@x
        elif method == 'tf_norm':
            result = (self.A/np.linalg.norm(self.A,axis=0)).T@x
        elif method == 'cossim':
            result = (self.A/np.linalg.norm(self.A,axis=0)).T@(x/np.linalg.norm(x))
        elif method == 'tfidf':
            result = self.B.T@x
        else:
            raise f"method {method} invalid, try ['tf','tf_norm','cossim','tfidf']"         
        return pd.DataFrame({'category':self.cat2index.keys(),'score':result.T[0]}).sort_values('score',ascending=False)
        
    def transform(self, text, toarray=True):
        """
        Função transform
        Retorna a representação bag of words de um texto ou um vetor de um texto
        Realiza por padrão a normalização (para minúsculo e retira acentos) e a tokenização por espaço
        Argumentos:
        text = texto a transformar
        toarray = por padrão é o valor de retorno, se False retorno no formato de matriz esparsa
        """
        if type(text)==type(''):        
            bagofwords = self.text2vec.transform([text])
        else: 
            bagofwords = self.text2vec.transform(text)
        return bagofwords.toarray() if toarray else bagofwords 
            
        
    def cos_similarity(self, text1, text2):
        vector01 = self.transform(text1)
        vector02 = self.transform(text2)
        return vector01@vector02/(np.linalg.norm(vector01) * np.linalg.norm(vector02))

    def vocabulary(self):
        """
        Função vocabulary
        retorna um dicionário com o vocabulário gerado e seu índice
        """
        return self.text2vec.vocabulary_

    def token2index(self, token):
        return self.text2vec.vocabulary_.get(token)

    def index2token(self, index):
        return dict(zip(self.text2vec.vocabulary_.values(), self.text2vec.vocabulary_.keys())).get(index)

    #def most_similar(self, query):        
    
################################## Fuzzy Match ##################################

# Import module for data manipulation
import pandas as pd
# Import module for linear algebra
import numpy as np
# Import module for Fuzzy string matching
from fuzzywuzzy import fuzz, process
# Import module for regex
import re
# Import module for iteration
import itertools
# Import module for function development
from typing import Union, List, Tuple
# Import module for TF-IDF
from sklearn.feature_extraction.text import TfidfVectorizer,TfidfTransformer 
# Import module for cosine similarity
from sklearn.metrics.pairwise import cosine_similarity
# Import module for KNN
from sklearn.neighbors import NearestNeighbors

# String pre-processing
def preprocess_string(s):
    # Remove spaces between strings with one or two letters
    s = re.sub(r'[\d-]',"", s)
    return s

# String matching - TF-IDF
def build_vectorizer(
    clean: pd.Series,
    analyzer: str = 'word', 
    ngram_range: Tuple[int, int] = (1, 4), 
    n_neighbors: int = 1, 
    **kwargs
    ) -> Tuple:
    # Create vectorizer
    vectorizer = TfidfVectorizer(analyzer = analyzer, ngram_range = ngram_range, **kwargs)
    X = vectorizer.fit_transform(clean.values.astype('U'))

    # Fit nearest neighbors corpus
    nbrs = NearestNeighbors(n_neighbors = n_neighbors, metric = 'cosine').fit(X)
    return vectorizer, nbrs

# String matching - KNN
def tfidf_nn(
    messy, 
    clean, 
    vectorizer,
    nbrs,
    n_neighbors = 5, 
    **kwargs
    ):
    # Fit clean data and transform messy data
    input_vec = vectorizer.transform(messy)

    # Determine best possible matches
    distances, indices = nbrs.kneighbors(input_vec, n_neighbors = n_neighbors)
    nearest_values = [[y for i, y in enumerate(clean) if i in indices[l]] for l in range(len(indices))]
    return nearest_values, distances, vectorizer

# String matching - match fuzzy
def find_matches_fuzzy(
    row, 
    match_candidates, 
    limit = 5
    ):
    row_matches = process.extract(
        row, dict(enumerate(match_candidates)), 
        scorer = fuzz.token_sort_ratio, 
        limit = limit
        )
    result = [(row, match[0], match[1]) for match in row_matches]
    return result

# String matching - TF-IDF
def fuzzy_nn_match(
    messy,
    clean,
    column,
    col,
    limit = 5, **kwargs):
    nearest_values, _, vec = tfidf_nn(messy, clean, **kwargs)

    results = [find_matches_fuzzy(row, nearest_values[i], limit) for i, row in enumerate(messy)]
    df = pd.DataFrame(itertools.chain.from_iterable(results),
        columns = [column, col, 'Ratio']
        )
    df.rename(columns={'nm_item':'desc'}, inplace=True)
    return df, vec

# String matching - Fuzzy
def fuzzy_tf_idf(
    df: pd.DataFrame,
    column: str,
    clean: pd.Series,
    mapping_df: pd.DataFrame,
    col: str,
    analyzer: str = 'word',
    ngram_range: Tuple[int, int] = (1, 3),
    **kwargs
    ) -> pd.Series:
    # Create vectorizer
    clean = clean.drop_duplicates().reset_index(drop = True)
    messy_prep = df[column].drop_duplicates().dropna().reset_index(drop = True).astype(str)
    #messy = messy_prep.apply(preprocess_string)
    result, vec = fuzzy_nn_match(messy = messy_prep, clean = clean, column = column, col = col, n_neighbors = 15, **kwargs)
    # Map value from messy to clean
    return result

################################## Importar Neodatabases ##################################
def connect_azure_training(database='datascience-neogrid'):
    server = 'datascience-neogrid.database.windows.net'
    username='rerodrigues'
    password = 'Analytics2021'
    driver = 'ODBC Driver 17 for SQL Server'
    string_conexao = 'DRIVER='+driver+';SERVER='+server+';PORT=1433;DATABASE='+database+';UID='+username+';PWD='+ password+';'    
    return pyodbc.connect(string_conexao)


################################## Configurações de Página ##################################
st.set_page_config(page_title="Saneamento Ativo",
                   page_icon=":bar_chart:",
                   layout="wide")

################################## Função para Obter Dados ##################################
#@st.cache(allow_output_mutation=True)
#def get_df():
#    with connect_azure_training() as conn:
#        df = pd.read_sql('SELECT * FROM sandbox.tbl_saneamento_teste', conn)
#        product = pd.read_sql('SELECT id_product, nm_product FROM tbl_product', conn)
#        df = df.merge(product, how='left')
#    return df, product

#df, product = get_df()
################################## Definir índice utilizado no ILOC ##################################
if 'count' not in st.session_state:
    st.session_state.count = 0
    st.session_state.dict_saneados = {}
    st.session_state.n_saneados = 0 
    st.session_state.n_saneados_id = []

################################## Função para Classificar Dados ##################################
@st.cache(allow_output_mutation=True)
def get_df():
    with connect_azure_training() as conn:
      # Get Data
      df = pd.read_sql('SELECT * FROM SANEAMENTO.ab_com_vendor', conn)
      product = pd.read_sql('SELECT id_product, nm_product FROM tbl_product', conn)
      df = df.merge(product, how='left')
      st.session_state.n_saneados = df[df.id_product.isna()].shape[0]
      st.session_state.n_saneados_id = df[df.id_product.isna()].id_item.tolist()
      return df, product

@st.experimental_memo
def clf():   
    # Check for NULLs
    if st.session_state.n_saneados > 0:
        messy = df[df.id_product.isna()].reset_index().drop(columns='index')
        clean = df[~df.id_product.isna()]
        vectorizer, nbrs = build_vectorizer(clean=clean.nm_item,n_neighbors=5)

        cat = NeoNLP()
        cat.build_model(clean.nm_item)
        cat.fit(description=clean.nm_item, classification=clean.nm_product)

        return vectorizer, nbrs, clean, messy, cat
    else:
        return None, None, None, None

with st.spinner('Categorizando itens...'):
  vectorizer, nbrs, clean, messy, cat_clf = clf()
  df, product = get_df()


################################## Funções atribuídas aos botões ##################################
def next_item():
    st.session_state.count += 1
    #st.dataframe(pd.DataFrame(df.loc[st.session_state.count,['id_item','nm_item','nm_hierarchy_level_1','nm_hierarchy_level_2','nm_hierarchy_level_3']]).T)

def previous_item():
    st.session_state.count -= 1
    #st.dataframe(pd.DataFrame(df.loc[st.session_state.count,['id_item','nm_item','nm_hierarchy_level_1','nm_hierarchy_level_2','nm_hierarchy_level_3']]).T)

def update_data(id_item, nm_product, cat):
    
    # Get id_product
    st.session_state.dict_saneados[str(id_item)] = [nm_product,cat]
    st.session_state.n_saneados = len([x for x in st.session_state.n_saneados_id if str(x) not in st.session_state.dict_saneados.keys()])
    
    id_product = product[product.nm_product == nm_product].id_product.tolist()[0]
    messy.loc[st.session_state.count, 'id_product'] = id_product
    df.loc[df.id_item == id_item, 'nm_product'] = nm_product
    with connect_azure_training() as conn:
        cursor = conn.cursor()

        cursor.execute(f'''
                    UPDATE SANEAMENTO.ab_com_vendor
                    SET id_product = {id_product}
                    WHERE id_item = {id_item}
                    ''')
        conn.commit()


################################## Página principal (bloco superior) ##################################
def main_page():
    if st.session_state.n_saneados > 0:
        l, r = st.columns([4,1])
        with l:
            st.header('Saneamento Ativo')
            st.dataframe(pd.DataFrame(messy.loc[st.session_state.count,['id_item','id_product','nm_item','nm_hierarchy_level_1','nm_hierarchy_level_2','nm_hierarchy_level_3']]).T)
        with r:
            st.header('Categorização')
            cat = cat_clf.predict(text=messy.nm_item.tolist()[st.session_state.count], method='tfidf')[0]
            lst = [x for x in product.nm_product.tolist() if x != cat]
            lst.insert(0, cat)
            x = st.selectbox(label='Selecione a categoria do item:', options=lst)



        ################################## Página principal (linha de botôes) ##################################
        a,b,c,d,e,f,g,h,i,j = st.columns([1,1,1,1,1,1,1,0.5,0.5,0.5])

        with h:
            if st.button('Atualizar'):
                update_data(nm_product=x, id_item=messy.loc[st.session_state.count].id_item, cat=cat)
        with j:
            st.button('Next', on_click=next_item)
        with i:
            if st.session_state.count > 0:
                st.button('Previous', on_click=previous_item)

        "---"
        ################################## Página principal (bloco inferior) ##################################
        l, m, r = st.columns(3)
        with l:
            st.subheader('Fuzzy Match - Sugestões')
            df_result = fuzzy_tf_idf(df=pd.DataFrame(messy.loc[st.session_state.count,:]).T,clean=clean.nm_item,column='nm_item',col='Result',mapping_df=clean,nbrs=nbrs,vectorizer=vectorizer)
            final = df_result.merge(df[['nm_item','id_product']], left_on='Result', right_on='nm_item').merge(product)
            st.table(final.loc[final.desc == messy.loc[st.session_state.count, 'nm_item'], ['Result','nm_product','Ratio']].sort_values('Ratio', ascending=False)) 
        with m:
            st.subheader('Machine Learning - Sugestões')
            pred = pd.DataFrame({'tf': cat_clf.predict(text=messy.nm_item.tolist()[st.session_state.count], method='tf'),
                                'tf_norm': cat_clf.predict(text=messy.nm_item.tolist()[st.session_state.count], method='tf_norm'),
                                'cossim': cat_clf.predict(text=messy.nm_item.tolist()[st.session_state.count], method='cossim'),
                                'tfidf': cat_clf.predict(text=messy.nm_item.tolist()[st.session_state.count], method='tfidf')}, index = ['CLASS']).T
            st.table(pred)
        with r:
            st.subheader('Itens com mesmo GTIN')
            st.table(df.loc[df.gtin.isin(df.loc[df.id_item == messy.loc[st.session_state.count, 'id_item']].gtin.tolist()), ['id_item','nm_item','gtin','nm_product']])
    else:
        st.markdown('Nenhum item para sanear!')

def second_page():
    col1, col2, col3 = st.columns(3)
    col1.metric("Não Saneados", st.session_state.n_saneados)
    col2.metric("Saneados", len(st.session_state.dict_saneados))
    col3.metric("Categorização Correta", sum([x[0] == x[1] for x in st.session_state.dict_saneados.values()]))

page_names_to_funcs = {
    "Saneamento": main_page,
    "Acompanhamento": second_page
}
selected_page = st.sidebar.radio("Selecione a página", page_names_to_funcs.keys())
page_names_to_funcs[selected_page]()

# sqlcmd = """
#     SELECT 
# 	raiz_cnpj,
# 	nm_manufatura,
# 	nm_product
# FROM 
# 	SANEAMENTO.tbl_manufatura tm 
# INNER JOIN 
# 	SANEAMENTO.tbl_marcas tm2 ON tm.id_manufatura = tm2.id_manufatura 
# INNER JOIN 
# 	SANEAMENTO.ab_com_vendor acv ON tm2.id_brand = acv.id_brand 
# INNER JOIN 
# 	dbo.tbl_product tp ON acv.id_product = tp.id_product 
#          """


# with connect_azure_training() as conn:
#     df = pd.read_sql(sqlcmd, conn)


# x = st.text_input("Digite um CNPJ ou razão social:")

# # eval

# if df[df.raiz_cnpj == x[:8]].shape[0] > 0:
#     opts = df[df.raiz_cnpj == x[:8]].nm_manufatura.unique().tolist()
#     opts.insert(0, 000)
    
#     st.success('Encontramos as seguintes manufaturas')
#     manuf = st.selectbox(label='Selecione a manufatura de interesse:', options=opts, format_func=lambda x: '<select>' if x == 000 else x)
#     if manuf:
#         opts_m = df[df.nm_manufatura == manuf].nm_product.unique().tolist()
#         opts_m.insert(0, '')
#         st.selectbox(label='Selecione a categoria de interesse:', options=opts_m, format_func=lambda x: '<select>' if x == '' else x)
# elif x != '':
#     st.warning('CNPJ não encontrado')

# if st.button("Reset"):
#     pyautogui.hotkey("ctrl","F5")

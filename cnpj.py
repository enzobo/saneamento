from matplotlib.pyplot import show
import pandas as pd  # pip install pandas openpyxl
import numpy as np
import plotly.express as px
import pyautogui
from fuzzywuzzy import fuzz, process
import streamlit as st
from streamlit.legacy_caching import clear_cache


################################## Importar Neodatabases ##################################
def connect_azure_training(database='datascience-neogrid'):
    server = 'datascience-neogrid.database.windows.net'
    username='rerodrigues'
    password = 'Analytics2021'
    driver = 'ODBC Driver 17 for SQL Server'
    string_conexao = 'DRIVER='+driver+';SERVER='+server+';PORT=1433;DATABASE='+database+';UID='+username+';PWD='+ password+';'    
    return pyodbc.connect(string_conexao)
st.set_page_config(page_title="CNPJ Insights",
                   page_icon=":bar_chart:",
                   layout="wide")

if 'count' not in st.session_state:
    st.session_state.count = 0
    
sqlcmd = """SELECT  id_item,
                    id_source,
                    nm_marca, 
                    raiz_cnpj,
                    nm_manufatura,
                    abv.id_product,
                    nm_category 
            FROM SANEAMENTO.tbl_manufatura tm 
            INNER JOIN SANEAMENTO.tbl_marcas tma ON tm.id_manufatura = tma.id_manufatura
            INNER JOIN SANEAMENTO.ab_com_vendor abv ON abv.id_brand = tma.id_brand 
            INNER JOIN dbo.dim_tag tag ON tag.id_product = abv.id_product  
            WHERE nm_manufatura IS NOT NULL"""

@st.cache()
def get_df():
    with connect_azure_training() as conn:
        manufatura = pd.read_sql(sqlcmd, conn)
        map_manuf_marca = dict(zip(manufatura.nm_marca, manufatura.nm_manufatura))
        lst = manufatura.nm_manufatura.unique().tolist() + manufatura.nm_marca.unique().tolist()
        return manufatura, map_manuf_marca, lst

def generate_plt(dataframe, column, percentage=True, all=False):
    if not all:
        plt = px.line(dataframe, 
                    x="cd_month", 
                    y=column, 
                    color="agrupamento", 
                    line_dash="agrupamento",
                    category_orders={'agrupamento':['Varejo','Outros']},
                    labels = {"agrupamento":"", "cd_month":""}, 
                    markers = True, 
                    title = column.split('_')[1].capitalize() + ' %' if percentage else column.split('_')[1].capitalize(), 
                    color_discrete_map = {'Outros':'#808080','Varejo':'#00263C'})

        plt.update_traces(textposition="bottom right", hovertemplate=None)
        plt.update_xaxes(dtick="M1")
        plt.update_layout(yaxis_title=None, 
                          xaxis_tickformat = '%b-%Y', 
                          hovermode="x unified", 
                          paper_bgcolor='rgba(0,0,0,0)', 
                          plot_bgcolor='rgba(0,0,0,0)', 
                          yaxis = dict(tickfont = dict(size=15)),
                          xaxis = dict(tickfont = dict(size=15)))
    else:
        plt = px.line(dataframe, 
                    x="cd_month", 
                    y=column, 
                    labels = {"cd_month":""}, 
                    markers = True, 
                    title = column.split('_')[1].capitalize() + ' %' if percentage else column.split('_')[1].capitalize(),
                    color_discrete_sequence = ['#00263C'])

        plt.update_traces(textposition="bottom right", hovertemplate=None)
        plt.update_xaxes(dtick="M1")
        plt.update_layout(yaxis_title=None, 
                          xaxis_tickformat = '%b', 
                          hovermode="x unified", 
                          paper_bgcolor='rgba(0,0,0,0)', 
                          plot_bgcolor='rgba(0,0,0,0)',
                          yaxis = dict(tickfont = dict(size=15)),
                          xaxis = dict(tickfont = dict(size=15)))
    return plt

def add_count():
    st.session_state.count = 1

def reset_count():
    st.session_state.count = 0

@st.cache()
def check_cnpj(cnpj, dataframe):
    if cnpj.isnumeric() and dataframe[dataframe.raiz_cnpj == cnpj[:8]].shape[0] > 0:
        choices = dataframe[dataframe.raiz_cnpj == cnpj[:8]].nm_manufatura.unique().tolist()
    elif not cnpj.isnumeric():
        choices = list(dict.fromkeys([map_manuf_marca.get(name, name) for name, score in process.extract(query=cnpj, choices=lst, scorer=fuzz.partial_token_sort_ratio, limit=10)]))
    else:
        choices = []
    return choices

@st.cache(allow_output_mutation=True)
def generate_category(dataframe, manufatura):
    agrup = dataframe[dataframe.nm_manufatura == manufatura].groupby('nm_category').agg({'id_product':'unique', 'id_source': 'unique'}).reset_index()
    category = dataframe[dataframe.nm_manufatura == manufatura].nm_category.unique().tolist()
    return agrup, category

@st.cache()
def generate_fact(dataframe_agrupado, dataframe_manufaturas, categoria, manufatura):
    product_list = dataframe_agrupado[dataframe_agrupado.nm_category == categoria].id_product.tolist()[0]
    source_list = dataframe_agrupado[dataframe_agrupado.nm_category == categoria].id_source.tolist()[0]
    item_list = dataframe_manufaturas[(dataframe_manufaturas.id_product.isin(product_list))&(dataframe_manufaturas.id_source.isin(source_list))].id_item.tolist()
    
    with connect_azure_training() as conn:
        fato = pd.read_sql("SELECT * FROM BENCHMARK.tbl_dash WHERE id_item IN {itens} AND cd_month LIKE '2022%'".format(itens=tuple(item_list)), conn)
    
    df = fato.merge(dataframe_manufaturas[['id_item', 'nm_manufatura']])
    dict_agg = {"vl_cnt_rows": "sum",
                "vl_cnt_of_stock": "sum",
                "vl_qty_stock": "sum",
                "vl_qty_sold": "sum",
                "vl_osa": "mean"}

    df['agrupamento'] = ['Manufatura' if x == manufatura else 'Outros' for x in df['nm_manufatura']]
    agrupado_mensal = df.groupby(['cd_month', 'agrupamento']).agg(dict_agg).reset_index()
    agrupado_mensal['vl_cobertura'] = agrupado_mensal['vl_qty_stock']/(agrupado_mensal['vl_qty_sold'])
    agrupado_mensal['vl_giro'] = (agrupado_mensal['vl_qty_sold']/agrupado_mensal['vl_qty_stock'])*365
    agrupado_mensal['vl_ruptura'] = (agrupado_mensal['vl_cnt_of_stock']/agrupado_mensal['vl_cnt_rows'])*100
    
    return agrupado_mensal

# Obtendo DataFrame
manufatura, map_manuf_marca, lst = get_df()

cnpj_razao = st.text_input("Digite um CNPJ ou razão social:", on_change=reset_count)

if cnpj_razao:
    choices = check_cnpj(cnpj=cnpj_razao, dataframe=manufatura)
    if choices:
        st.success('Encontramos as seguintes manufaturas')
        #choices = list(dict.fromkeys([map_manuf_marca.get(name, name) for name, score in process.extract(query=x, choices=lst, scorer=fuzz.partial_token_sort_ratio, limit=10)]))
        manuf = st.selectbox(label='Selecione a manufatura de interesse:', options=choices)
    
        if manuf:
            agrup, category = generate_category(dataframe=manufatura, manufatura=manuf)
            category_selected = st.multiselect(label='Selecione até 5 categorias:', options=category)
            if category_selected:
                generate_analysis = st.button('Gerar análise', on_click=add_count)
                if st.session_state.count > 0:
                    category_analysis = st.selectbox(label='Categoria Analisada:', options=category_selected)
                    agrupado_mensal = generate_fact(dataframe_manufaturas=manufatura, dataframe_agrupado=agrup, categoria=category_analysis, manufatura=manuf)

                    left_column, right_column = st.columns(2)
                    with left_column:
                        st.plotly_chart(generate_plt(dataframe=agrupado_mensal, column='vl_ruptura'), use_container_width=True)
                        st.plotly_chart(generate_plt(dataframe=agrupado_mensal, column='vl_giro', percentage=False), use_container_width=True)
                    with right_column:
                        st.plotly_chart(generate_plt(dataframe=agrupado_mensal, column='vl_osa'), use_container_width=True)
                        st.plotly_chart(generate_plt(dataframe=agrupado_mensal, column='vl_cobertura', percentage=False), use_container_width=True)
    else:
        st.warning('Nenhuma manufatura encontrada')

if st.button("Reset"):
    clear_cache()
    pyautogui.hotkey("ctrl","F5")

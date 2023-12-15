from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from cytoolz import compose, curry

from pdf import generate_faiss_idx

vectorstore = generate_faiss_idx('pdfs/apple.pdf')
retriever = vectorstore.as_retriever()

query_template = """Answer the question based only on the following context:
{context}

Question: {question}
"""

prompt = ChatPromptTemplate.from_template(query_template)

model = ChatOpenAI(model='gpt-4')

chain = (
    {'context': retriever, 'question': RunnablePassthrough()}
    | prompt
    | model
    | StrOutputParser()
)

def template_gen(metric_name: str, year: int) -> str:
    return f'''What are {metric_name} for the year {year}? Please return everything row by row in a pipe separated string, like so:
    asset_name1|value\n
    asset_name2|value\n
    ...,
    asset_name_N|value\n
    '''

def get_table(metric_name, year) -> pd.DataFrame:
    return compose(curry(load_into_dataframe, column_names=['metric', 'value']),
                   chain.invoke,
                   curry(template_gen, year=year))(metric_name)


current_assets = get_table('current asset values', 2018)

balance_sheet = get_table('the balance sheet values', 2018)

result = chain.invoke('''What are the balance sheet values for the year 2018? Please return a json file

In the case of subtables, like:

Subtable name  (no value)
  metric name  value

please make the subtable name a higher level than the metrics below it.

For headers, like ASSETS, please return them as a higher level as well

Please return all of the values, including both assets and liabilities

For example:

{'assets':
  {'subheader': {'metric': 'value', ..., 'metric': 'value}},
  {'subheader': {'metric': 'value', ..., 'metric': 'value}},
  {'subheader': {'metric': 'value', ..., 'metric': 'value}},
 'liabilities':
  {'subheader': {'metric': 'value', ..., 'metric': 'value}},
  {'subheader': {'metric': 'value', ..., 'metric': 'value}},
  {'subheader': {'metric': 'value', ..., 'metric': 'value}},
}
''')



def load_into_dataframe(csv_string, column_names):
    data = []
    for line in csv_string.strip().split('\n'):
        line = line.replace('$', '')
        metric, value = line.split('|', 1)
        value = value.replace(',', '').strip()
        if value.isdigit():
            data.append((metric.strip(), int(value)))
    df = pd.DataFrame(data, columns=column_names)
    return df

column_names = ['metric', 'value']
df = load_into_dataframe(result, column_names)
print(df)

import streamlit as st
from utils import * 
from deeplx import *

OPTIONS = ['gpt-3.5-turbo','gpt-4-turbo-preview','claude-3-haiku','claude-3-sonnet','claude-3-opus']

st.set_page_config(layout='wide')
st.title('论文润色降重')

st.sidebar.title('提示文本')
prompt = st.sidebar.text_area('提示文本',value=None,height=300)

if prompt is None:
    prompt = f"你是一位精通各领域论文写作的院士级教授，为了论文的降重需求，你将根据我提供的文本进行改写和修正，\
        改写后的文本应该符合原文意思、具有逻辑性和阅读流畅且不应与原文有过多重复，文本中出现的人名、单位和专业术语不需要进行翻译，\
        你会使用中文回复我改写后的文本内容，（永远不要对文本进行解释、扩写和提供建议！）"

windows = st.number_input('同时进行模型数',value=1)

input_text = Text_area(label='输入文本',label_visibility='visible').run()
token_text = Input_area(label='用户密钥',label_visibility='visible').run()

result_areas = [Text_area(label=f'模型输出{i}') for i in range(windows)]
select_boxs = [Select_box(label=f'模型选择{i}',options=OPTIONS,default=OPTIONS[0]) for i in range(windows)]
object_dict = {i:{'text':result_areas[i],'select':select_boxs[i]} for i in range(windows)}


Columns(object_dict,text=input_text,token = token_text, windows=windows, prompt=prompt).run()
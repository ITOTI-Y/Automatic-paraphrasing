import streamlit as st
from deeplx import *
st.title('论文润色降重')

text = st.text_area('输入要降重的文本：',height=200)
token = st.text_input('输入密钥')
model_selectbox = st.selectbox('选择模型',['gpt-3.5-turbo','gpt-4-turbo-preview','claude-3-haiku','claude-3-sonnet','claude-3-opus'])
model_input = st.text_input('自定义模型',value= None)

model = model_input or model_selectbox

prompt = st.sidebar.text_input('覆写提示', value= None)

if 'result' not in st.session_state:
    st.session_state.result = None

if st.button('开始降重'):
    st.session_state.result = ai_polishing(text,token, model = model, prompt = prompt)

if st.session_state.result:
    lines1 = len(st.session_state.result) // 40 + 1
    height1 = min(max(lines1 * 28, 150),500)
    st.session_state.result = st.text_area('润色结果:',value=st.session_state.result,height=height1)

if st.checkbox('自动AI转翻译'):
    if st.session_state.result:
        respond = double_translate(st.session_state.result)
        lines2 = len(respond[1]) // 40 + 1
        height2 = min(max(lines2 * 28, 150),500)
        st.text_area('翻译结果:',value=respond[1],height=height2)
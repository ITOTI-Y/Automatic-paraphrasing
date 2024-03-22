import streamlit as st
from deeplx import *
from typing import Literal
import numpy as np
import concurrent.futures

def height_num(text:str, lines:int=40):
    if text is None:
        text_len = 0
    else:
        text_len = len(text)
    height = min(max(text_len // lines + 1, 1) * 28, 1000)
    return height

class Text_area:
    def __init__(self,label:str, text:str=None, height:int=200,label_visibility:Literal['visible','hidden','collapsed']='visible'):
        self.label = label
        self.text = text
        self.height = height
        self.label_visibility = label_visibility
        self.key_id = str(hash(label))
        if self.key_id not in st.session_state:
            st.session_state[self.key_id] = None
        height = height_num(text) if text else height

    def run(self):
        self.result = st.text_area(label=self.label,value=st.session_state[self.key_id], height=self.height,label_visibility=self.label_visibility)
        st.session_state[self.key_id] = self.result
        return self.result


class Input_area:
    def __init__(self,label:str, text:str=None,label_visibility:Literal['visible','hidden','collapsed']='visble'):
        self.label = label
        self.text = text
    
    def run(self):
        return st.text_input(label=self.label,value=self.text)
    
class Select_box:
    def __init__(self,label:str,options:list[str],default:str):
        self.label = label
        self.options = options
        self.default = default
    
    def run(self):
        return st.selectbox(label=self.label,options=self.options,index=self.options.index(self.default))

class Button:
    def __init__(self,label:str):
        self.label = label
    
    def run(self):
        return st.button(label=self.label)
    
class Check_box:
    def __init__(self,label:str):
        self.label = label
    
    def run(self):
        return st.checkbox(label=self.label)
    
class Side_bar:
    def __init__(self):
        pass

class Columns:
    def __init__(self,objects:dict['text':Text_area,'select':Select_box],text:str=None,token:str=None,prompt=None,windows:int=3):
        self.obj = objects
        self.text = text
        self.token = token
        self.prompt = prompt
        self.windows = windows

    def run(self):
        cols = st.columns(self.windows)
        models = [cols[i].selectbox(label=self.obj[i]['select'].label,options=self.obj[i]['select'].options, key=f'model_{i}') for i in range(self.windows)]
        if Button(label='提交').run():
            with concurrent.futures.ThreadPoolExecutor() as executor:
                ai_futures = [executor.submit(ai_polishing, text = self.text, token = self.token, model = models[i], prompt = self.prompt) for i in range(self.windows)]
                ai_results = [future.result()[0] for future in ai_futures]
                ai_data = [future.result()[1] for future in ai_futures]
                print(ai_data)
            max_height = max([height_num(ai_results[i],lines=90//self.windows) for i in range(self.windows)])
            st.session_state['max_height'] = max_height
            for i in range(self.windows):
                st.session_state[f'text_area_{i}'] = ai_results[i]
        if Check_box(label='显示模型输出').run():
            text_areas = [cols[i].text_area(label=self.obj[i]['text'].label, value=st.session_state.get(f'text_area_{i}', ''), key=f'text_area_{i}', height=st.session_state.get('max_height', 200)) for i in range(self.windows)]
        all_filled = all([st.session_state.get(f'text_area_{i}', '') for i in range(self.windows)])
        if all_filled:
            with concurrent.futures.ThreadPoolExecutor() as executor:
                deepl_futures = [executor.submit(double_translate, st.session_state[f'text_area_{i}'],) for i in range(self.windows)]
                deepl_results = [future.result() for future in deepl_futures]
                for i in range(self.windows):
                    st.session_state[f'deepl_area_{i}'] = deepl_results[i]
            for i in range(self.windows):
                cols[i].text_area(label=f'结果{i}', value=st.session_state.get(f'deepl_area_{i}', ''), key=f'deepl_area_{i}', height=st.session_state.get('max_height', 200))

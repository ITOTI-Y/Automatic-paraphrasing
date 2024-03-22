import requests

def translate_text(text, source_language="ZH", target_language="EN"):
    url = "https://api.deeplx.org/translate"
    data = {
        "text": text,
        "source_lang": source_language,
        "target_lang": target_language
    }
    response = requests.post(url, json=data)
    return response.json()["data"]

def double_translate(text, source_language="ZH", target_language="EN"):
    one_step = translate_text(text, source_language, target_language)
    two_step = translate_text(one_step, target_language, source_language)
    return two_step

def ai_polishing(text,token,model = "claude-3-haiku-patch",prompt = None):
    url = "https://aigptx.top/v1/chat/completions"
    headers = {"Authorization": f"Bearer {token}"}
    # if prompt is None:
    #     prompt = f"你是一位精通各领域论文写作的院士级教授，为了论文的降重需求，你将根据我提供的文本进行改写和修正，\
    #         改写后的文本应该符合原文意思、具有逻辑性和阅读流畅且不应与原文有过多重复，文本中出现的人名、单位和专业术语不需要进行翻译，\
    #         你会使用中文回复我改写后的文本内容，（永远不要对文本进行解释、扩写和提供建议！）"
    data = {
        "model":f"{model}",
        "messages":[
            {
                "role":"system",
                "content":f"{prompt}"
            },
            {
                "role":"user",
                "content":f"{text}"
            }
        ]
    }
    response = requests.post(url, headers=headers, json=data).json()['choices'][0]['message']['content']
    return response,data
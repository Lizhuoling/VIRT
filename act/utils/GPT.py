import requests
import json
import time
import base64
import hashlib
from mimetypes import guess_type


app_infos = {
            "appId": "483b93faf68d43d2b7e46d2d7a889c1b",
            "appSecret": "iVnpRkueXvPgdQG8SVx6sXeElZhdikFj",
            }


def local_image_to_data_url(image_path):
    # Guess the MIME type of the image based on the file extension
    mime_type, _ = guess_type(image_path)
    if mime_type is None:
        mime_type = 'application/octet-stream'  # Default MIME type if none is found

    # Read and encode the image file
    with open(image_path, "rb") as image_file:
        base64_encoded_data = base64.b64encode(image_file.read()).decode('utf-8')

    # Construct the data URL
    return f"data:{mime_type};base64,{base64_encoded_data}"

def get_access_token(app_id, app_secret):
    timestamp = str(int(time.time()))

    _sign = "appId=%s&appSecret=%s&timestamp" % (app_id, app_secret) + "=" + timestamp

    sign = hashlib.md5(_sign.encode('utf-8')).hexdigest()

    headers = {
        'x-open-sign': sign,
        'x-open-grantType': 'authorization',
        'x-open-timestamp': timestamp,
        'x-open-appId': app_id,
        'Content-Type': 'application/json',
    }

    res = requests.post('https://openapi.seewo.com/api/v1/token/access', headers=headers, data='{"query":"","variables":{}}')

    return res.json()['body']['accessToken']

def run_gpt_api(app_infos, 
                user_prompt="你是谁？",
                system_prompt="你是万能的回答助手",
                gpt_model="OPENAI_GPT_4_O_PREVIEW",
                max_tokens=2048,
                image_path=None):
    
    """
    gpt_model support API:
        GOOGLE_GEMINI_1_PRO
        GOOGLE_GEMINI_1_PRO_VISION
        GOOGLE_GEMINI_1_5_PRO_PREVIEW_0409
        GOOGLE_GEMINI_1_5_FLASH_PREVIEW_0514
        OPENAI_GPT_35
        OPENAI_GPT_35_16K
        OPENAI_GPT_4
        OPENAI_GPT_4_32K
        OPENAI_GPT_4_TURBO
        OPENAI_GPT_4_VISION_PREVIEW
        OPENAI_GPT_4_O_PREVIEW
    """

    app_id = app_infos['appId']
    app_secret = app_infos['appSecret']

    access_token = get_access_token(app_id, app_secret)

    url = "http://ai-open.test.seewo.com/api/v1/nlp/raw/chat/completion?model={}".format(gpt_model)

    headers = {
        "Content-Type": "application/json",
        "x-open-accessToken": access_token,
        "x-open-appId": app_id,
        "x-open-userId": "EmbodyAI-test"
    }

    if image_path is not None:
        image_url = local_image_to_data_url(image_path)
        user_content = [
            {
                "type": "image_url",
                "image_url": {
                    "url": image_url
                }
            },
            {
                "type": "text",
                "text": user_prompt
            }
        ]
    else:
        user_content = [
            {
                "type": "text",
                "text": user_prompt
            }
        ]

    data = {
        "messages": [
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": user_content
            }
        ],
        "max_tokens": max_tokens
    }

    response = requests.post(url, headers=headers, data=json.dumps(data))
    response_info = response.json()

    return response_info['choices'][0]['message']['content'], response_info

if __name__ == "__main__":

    response, response_info = run_gpt_api(app_infos=app_infos, 
                user_prompt="Due to the multi-modal nature of trajectories, robotic manipulation is highly ambiguous and often requires instructions to specify the task content. In this work, we highlight that vision instruction, compared with text instruction, is more suitable for use as it describes manipulation details more clearly and is easier to understand by policies. Following this insight and also borrowing inspirations from cognitive science, we propose the robotic imagery paradigm that supports pre-training on large-scale robotic data without any human annotation. In addition, mimicking the mechanism of human eye gaze, we devise a robotic gaze strategy that guides subsequent actions and concentrates the attention of the policy on the manipulated object. Based on these two proposed techniques, a fully Transformer-based policy named VIRT is developed. We design extensive tasks using both a real robot and simulation environments to validate the effectiveness of VIRT. The results indicate that VIRT can complete very competitive tasks like ``opening the lid of a tightly sealed bottle'', and the proposed techniques boost the success rates of the baseline policy on some challenging tasks from 0\% to more than 70\%.",
                system_prompt="You are an expert in academic paper writting. Please help me express the following content in a more professional, concise, and academic tone.",
                gpt_model="OPENAI_GPT_4_O_PREVIEW",
                image_path=None)

    print(response)


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
                user_prompt="I am doing a project about demonstration learning based robotic manipulation. I design a fully Transformer policy consisting of encoders and decoders. For \
                    the input to the policy, I input not only robot status information, observation images of three views, and also vision instruction. The vision instruction is used to prompt \
                    the policy what to do. We believe text information is insufficient to describe the task because the policy does not understand natural language and the data is not sufficient to \
                    make it understand it. Therefore, during training, we use an external detector to recognize the object to manipulate and crop the image region of this target object. Then, we resize \
                    the cropped image to the original image resolution and input it to the policy as vision instruction, which informs the policy about what object to manipulate. We find that the vision \
                    instruction is very crucial for successful manipulation. If we replace the vision instruction as a text sentence that describes what to manipulate, the trained policy cannot learn \
                    how to complete the desired task. However, this story is too naive to write in a paper. I need you to build a mathematical theory for this story \
                    about why the vision instruction is crucial. The story should be written in a mathematical way and it can be as complex as possible.",
                system_prompt="You are an expert in mathematics and information theory.",
                gpt_model="OPENAI_GPT_4_O_PREVIEW",
                image_path=None)

    print(response)


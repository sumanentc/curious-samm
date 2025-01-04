import json
import os

import streamlit as st
import io

from langchain.chains.llm import LLMChain
from langchain.memory import ConversationBufferMemory
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, HumanMessagePromptTemplate
from langchain_openai import ChatOpenAI
from openai import OpenAI
from pydantic import BaseModel, Field
from pydub import AudioSegment
import base64

from prompt import GET_ANSWER_PROMPT


class ValidQuestion(BaseModel):
    valid_question: bool = Field(description="question asked by user is valid or not")


def check_valid_question(prompt: str, json_result=True,):
    llm = ChatOpenAI(temperature=0, model_name='gpt-4o-mini', openai_api_key=st.secrets["openai_api_key"],
                     max_retries=2,
                     max_tokens=100)
    messages = [
        SystemMessage(
            content="You are an expert Teacher, who loves to teach/answer questions."
        ),
        HumanMessage(
            content=prompt
        ),
    ]
    try:
        # Set up a parser + inject instructions into the prompt template.
        parser = JsonOutputParser(pydantic_object=ValidQuestion)
        response = llm.invoke(messages)
    except Exception as e:
        print(f'Not able to parse the LLM response: {e}')
        return json.loads("")
    # print(response)
    return parser.parse(response.content) if json_result else response.content


def transcribe(audio_segment: AudioSegment) -> str:
    """
    Transcribe an audio segment using OpenAI's Whisper ASR system.
    Args:
        audio_segment (AudioSegment): The audio segment to transcribe.
    Returns:
        str: The transcribed text.
    """
    client = OpenAI(api_key=st.secrets["openai_api_key"])
    raw_buffer = io.BytesIO()
    raw_buffer.name = 'sample.wav'
    audio_segment.export(raw_buffer, format="wav")
    answer = None
    try:
        answer = client.audio.transcriptions.create(
            model="whisper-1",
            file=raw_buffer,
            prompt=f"The language is en. Do not transcribe if you think it is a noise or not a valid question."
                   f" Just say cannot understand. Don't make it up",
            response_format="text", temperature=0.1
        )
    except Exception as e:
        print(f'Not able to transcribe the audio input: {e}')
    finally:
        raw_buffer.close()
    return answer

# Function to encode the image
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def get_answer_from_image(prompt, image_path):
    client = OpenAI(api_key=st.secrets["openai_api_key"])
    # Getting the base64 string
    base64_image = encode_image(image_path)
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system",
             "content": [
                 {
                     "type": "text",
                     "text": GET_ANSWER_PROMPT,
                 }
             ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt,
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                    },
                ],
            }
        ],
        temperature=0,
        max_tokens=200,
    )
    # print(response.choices[0])
    return response.choices[0].message.content

def generate_voice_from_text(text):
    client = OpenAI(api_key=st.secrets["openai_api_key"])
    script_dir = os.path.dirname(__file__)
    rel_path = "audio/speech.mp3"
    abs_file_path = os.path.join(script_dir, rel_path)
    response = client.audio.speech.create(
        model="tts-1",
        voice="alloy",
        input=text,
    )
    response.write_to_file(abs_file_path)

import base64
import os
import threading
import time
from collections import deque
from typing import List

import av
import numpy as np
import pydub
import streamlit as st
from PIL import Image
from pydub import AudioSegment
from streamlit_webrtc import webrtc_streamer, WebRtcMode, AudioProcessorBase

from llm import transcribe, get_answer_from_image, check_valid_question, generate_voice_from_text
from prompt import get_valid_question_prompt

st.title("Curious 🧒🏻")
st.write('\n')

st.subheader('Seamless Interaction with Live Streams: Hands-Free, Voice-Powered Insights in Real-Time\n', divider='rainbow')
st.write('\n\n\n')
WIDTH = st.sidebar.select_slider('QUALITY (May reduce the speed)', list(range(150, 501, 50)))
width = WIDTH

lock = threading.Lock()
img_container = {"img": None}
# st_webrtc_logger = logging.getLogger("streamlit_webrtc")
# st_webrtc_logger.setLevel(logging.WARNING)


def autoplay_audio(file_path: str, audio):
    with open(file_path, "rb") as f:
        data = f.read()
        b64 = base64.b64encode(data).decode()
        md = f"""
            <audio controls autoplay="true">
            <source src="data:audio/mp3;base64,{b64}" type="audio/mp3">
            </audio>
            """
        audio.markdown(
            md,
            unsafe_allow_html=True,
        )


def get_answer(prompt):
    with lock:
        img = img_container["img"]
    if img is not None and prompt.strip():
        print(f'Getting Answer for question -> {prompt}')
        orig_h, orig_w = img.shape[0:2]
        input_image = Image.fromarray(img, ).resize((width, int(width * orig_h / orig_w)))
        input_image.save('images/input.png')
        # <-- absolute dir the image is in
        script_dir = os.path.dirname(__file__)
        rel_path = "images/input.png"
        abs_image_path = os.path.join(script_dir, rel_path)
        answer = get_answer_from_image(prompt,abs_image_path)
        if answer:
            generate_voice_from_text(answer)
        return answer


def video_frame_callback(frame):
    img = frame.to_ndarray(format="rgb24")
    print('inside video_frame_callback')
    with lock:
        img_container["img"] = img
    return frame


def frame_energy(frame):
    """
    Compute the energy of an audio frame.
    Args:
        frame (VideoTransformerBase.Frame): The audio frame to compute the energy of.
    Returns:
        float: The energy of the frame.
    """
    samples = np.frombuffer(frame.to_ndarray().tobytes(), dtype=np.int16).astype(np.int32)
    return np.sqrt(np.mean(samples ** 2))


def add_frame_to_chunk(audio_frame, sound_chunk):
    """
    Add an audio frame to a sound chunk.
    Args:
        audio_frame (VideoTransformerBase.Frame): The audio frame to add.
        sound_chunk (AudioSegment): The current sound chunk.
    Returns:
        AudioSegment: The updated sound chunk.
    """
    sound = pydub.AudioSegment(
        data=audio_frame.to_ndarray().tobytes(),
        sample_width=audio_frame.format.bytes,
        frame_rate=audio_frame.sample_rate,
        channels=len(audio_frame.layout.channels),
    )
    sound_chunk += sound
    return sound_chunk


def process_audio_frames(audio_frames, sound_chunk, silence_frames, energy_threshold):
    """
    Process a list of audio frames.
    Args:
        audio_frames (list[VideoTransformerBase.Frame]): The list of audio frames to process.
        sound_chunk (AudioSegment): The current sound chunk.
        silence_frames (int): The current number of silence frames.
        energy_threshold (int): The energy threshold to use for silence detection.
    Returns:
        tuple[AudioSegment, int]: The updated sound chunk and number of silence frames.
    """
    for audio_frame in audio_frames:
        sound_chunk = add_frame_to_chunk(audio_frame, sound_chunk)
        energy = frame_energy(audio_frame)
        if energy < energy_threshold:
            silence_frames += 1
        else:
            silence_frames = 0

    return sound_chunk, silence_frames


def __is_valid_question__(question):
    invalid_question_list = ['Do not transcribe', 'Thank you for watching', 'please post them in the comments',
                             'Thank you','Please leave a comment','It is a noise']
    valid = True
    print(f'question :: {question}')
    for invalid_question in invalid_question_list:
        if invalid_question in question:
            valid = False
            break
    if valid:
        valid_question_prompt = get_valid_question_prompt(question)
        response =  check_valid_question(valid_question_prompt)
        if response and response['valid_question']:
            valid = True
        else:
            valid = False
    # print(f'valid :: {valid}')
    return valid


def handle_silence(sound_chunk, silence_frames, silence_frames_threshold, text_output, answer, audio):
    """
    Handle silence in the audio stream.
    Args:
        sound_chunk (AudioSegment): The current sound chunk.
        silence_frames (int): The current number of silence frames.
        silence_frames_threshold (int): The silence frames threshold.
        text_output (st.empty): The Streamlit text output object.
        answer (st.empty): Setting the answer to the user question in this variable.
    Returns:
        tuple[AudioSegment, int]: The updated sound chunk and number of silence frames.
    """
    if silence_frames >= silence_frames_threshold:
        if len(sound_chunk) > 0:
            print('handle_silence .... ')
            question = transcribe(sound_chunk)
            question = question.replace(".", "")
            if question.strip():
                if __is_valid_question__(question):
                    text_output.write(f":rainbow[{question}] \n\n")
                    answer.write('')
                    audio.empty()
                    with st.spinner('Understanding your question wait for it...'):
                        question_answer = get_answer(question)
                    if question_answer:
                        answer.write(f" \n{question_answer} ")
                        script_dir = os.path.dirname(__file__)
                        rel_path = "audio/speech.mp3"
                        abs_file_path = os.path.join(script_dir, rel_path)
                        autoplay_audio(abs_file_path,audio)
            sound_chunk = pydub.AudioSegment.empty()
            silence_frames = 0

    return sound_chunk, silence_frames


def handle_queue_empty(sound_chunk, text_output):
    """
    Handle the case where the audio frame queue is empty.
    Args:
        sound_chunk (AudioSegment): The current sound chunk.
        text_output (st.empty): The Streamlit text output object.
    Returns:
        AudioSegment: The updated sound chunk.
    """
    if len(sound_chunk) > 0:
        print('handle_queue_empty .... ')
        text = transcribe(sound_chunk)
        text_output.write(text)
        sound_chunk = pydub.AudioSegment.empty()

    return sound_chunk


def app_sst(energy_threshold=2000, silence_frames_threshold=20):
    class AudioProcessor(AudioProcessorBase):
        frames_lock: threading.Lock
        frames: deque

        def __init__(self) -> None:
            self.frames_lock = threading.Lock()
            self.frames = deque([])

        async def recv_queued(self, frames: List[av.AudioFrame]) -> av.AudioFrame:
            with self.frames_lock:
                self.frames.extend(frames)

            # Return empty frames to be silent.
            new_frames = []
            for frame in frames:
                input_array = frame.to_ndarray()
                new_frame = av.AudioFrame.from_ndarray(
                    np.zeros(input_array.shape, dtype=input_array.dtype),
                    layout=frame.layout.name,
                )
                new_frame.sample_rate = frame.sample_rate
                new_frames.append(new_frame)

            return new_frames

    webrtc_ctx = webrtc_streamer(key="curious_child", mode=WebRtcMode.SENDRECV,
                                 video_frame_callback=video_frame_callback,
                                 media_stream_constraints={"video": True, "audio": True},
                                 audio_processor_factory=AudioProcessor, )

    if not webrtc_ctx.state.playing:
        return
    status_indicator = st.empty()
    question = st.empty()
    answer = st.empty()
    audio = st.empty()
    status_indicator.write("Loading...")
    sound_chunk = pydub.AudioSegment.empty()
    silence_frames = 0
    while True:
        if webrtc_ctx.audio_processor:
            audio_frames = []
            with webrtc_ctx.audio_processor.frames_lock:
                while len(webrtc_ctx.audio_processor.frames) > 0:
                    frame = webrtc_ctx.audio_processor.frames.popleft()
                    audio_frames.append(frame)
            print(f'Audio frames {len(audio_frames)}')
            if len(audio_frames) == 0:
                time.sleep(0.03)
                status_indicator.write("Not received any input yet.")
                continue

            status_indicator.write("Active. Say something!!")
            sound_chunk, silence_frames = process_audio_frames(audio_frames, sound_chunk, silence_frames,
                                                               energy_threshold)

            sound_chunk, silence_frames = handle_silence(sound_chunk, silence_frames, silence_frames_threshold,
                                                         question, answer, audio)
        else:
            status_indicator.write("Not able to hear you...")
            handle_queue_empty(sound_chunk, question)
            break


app_sst()

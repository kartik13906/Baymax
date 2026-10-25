from TTS.api import TTS

tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2")
tts.tts_to_file(
    text="Hello. I am Baymax, your personal healthcare companion. On a scale of 1 to 10, how would you rate your pain?",
    speaker_wav="voice_lab/baymax.wav",
    language="en",
    file_path="baymax_voice_output.wav",
)

from playsound3 import playsound
playsound("baymax_voice_output.wav")

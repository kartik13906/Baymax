# load all the functions
from voice_cloning.generation import *
import os

# 1. Provide the reference sound file path and the text you want Baymax to say
# Make sure the "voice labs" folder is in the same directory as this script!
sound_path = r"baymax.wav" 

speech_text = "Hello. I am Baymax, your personal healthcare companion. On a scale of 1 to 10, how would you rate your pain?"

# 2. Generate the cloned voice
print("Synthesizing Baymax's voice...")
generated_wav = speech_generator(
    voice_type="western", 
    sound_path=sound_path, 
    speech_text=speech_text
)

# 3. Play and save the sound with noise-reduction capabilities
print("Playing generated audio...")
play_sound(generated_wav)

print("Saving the audio file...")
# save the file with a Baymax-specific name
save_sound(generated_wav, filename="baymax_voice_output", noise_reduction=True) 

print("Voice cloned and saved successfully!")

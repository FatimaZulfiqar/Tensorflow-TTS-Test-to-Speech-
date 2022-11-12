import numpy as np
import soundfile as sf
import yaml
import tensorflow as tf

from tensorflow_tts.processor import LJSpeechProcessor
from tensorflow_tts.processor.ljspeech import LJSPEECH_SYMBOLS

from tensorflow_tts.configs import Tacotron2Config
from tensorflow_tts.configs import MelGANGeneratorConfig

from tensorflow_tts.models import TFTacotron2
from tensorflow_tts.models import TFMelGANGenerator

from IPython.display import Audio
print(tf.__version__)


# Load the TFLite model and allocate tensors.
interpreter = tf.lite.Interpreter(model_path='TFLite Model/tacotron2.tflite') #set path to tflite model here
interpreter.allocate_tensors()

# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Prepare input data.
def prepare_input(input_ids):
  return (tf.expand_dims(tf.convert_to_tensor(input_ids, dtype=tf.int32), 0),
          tf.convert_to_tensor([len(input_ids)], tf.int32),
          tf.convert_to_tensor([0], dtype=tf.int32))
  
# Test the model on random input data.
def infer(input_text):
  processor = LJSpeechProcessor(None, symbols=LJSPEECH_SYMBOLS)
  input_ids = processor.text_to_sequence(input_text.lower())
  interpreter.resize_tensor_input(input_details[0]['index'], 
                                  [1, len(input_ids)])
  interpreter.allocate_tensors()
  input_data = prepare_input(input_ids)
  for i, detail in enumerate(input_details):
    print(detail)
    input_shape = detail['shape']
    interpreter.set_tensor(detail['index'], input_data[i])

  interpreter.invoke()
  # The function `get_tensor()` returns a copy of the tensor data.
  # Use `tensor()` in order to get a pointer to the tensor.
  return (interpreter.get_tensor(output_details[0]['index']),
          interpreter.get_tensor(output_details[1]['index']))
  

input_text = 'Recent research at Harvard has shown meditating\for as little as 8 weeks, can actually increase the grey matter in the\parts of the brain responsible for emotional regulation, and learning.'
decoder_output_tflite, mel_output_tflite = infer(input_text)
audio_before_tflite = melgan(decoder_output_tflite)[0, :, 0]
audio_after_tflite = melgan(mel_output_tflite)[0, :, 0]

Audio(data=audio_before_tflite, rate=22050)
Audio(data=audio_after_tflite, rate=22050)

# save to file
sf.write('./audio_before.wav', audio_before, 22050, "PCM_16")
sf.write('./audio_after.wav', audio_after, 22050, "PCM_16")

input_text = "I love TensorFlow Lite converted Tacotron 2."

decoder_output_tflite, mel_output_tflite = infer(input_text)
audio_before_tflite = melgan(decoder_output_tflite)[0, :, 0]
audio_after_tflite = melgan(mel_output_tflite)[0, :, 0]

Audio(data=audio_before_tflite, rate=22050)
Audio(data=audio_after_tflite, rate=22050)

# save to file
sf.write('./audio_before.wav', audio_before, 22050, "PCM_16")
sf.write('./audio_after.wav', audio_after, 22050, "PCM_16")
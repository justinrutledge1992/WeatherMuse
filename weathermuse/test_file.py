######################################################################
######################################################################

# Check environment compatibility
import torch
print("PyTorch version:", torch.__version__)
if (torch.__version__ != "2.1.0+cu121"):
    print("WARNING! Wrong version of torch installed; 2.1.0+cu121 is needed.")
print("CUDA available:", torch.cuda.is_available())
if (torch.cuda.is_available() != True):
    print("WARNING! CUDA is not available.")
print("CUDA version:", torch.version.cuda)
if (torch.version.cuda != "12.1"):
    print("WARNING! Wrong version of CUDA installed.")
print("\n")

######################################################################
######################################################################

# Test musicgen
from audiocraft.models import MusicGen
model = MusicGen.get_pretrained('facebook/musicgen-small')  # Options: 'small', 'medium', 'melody'
text_prompt = ["a serene piano melody similar to beethoven's midnight sonata"]
model.set_generation_params(duration=10)
generated_audio = model.generate(text_prompt, progress=True)

######################################################################
######################################################################

# Save the generated audio to a .wav file
import torchaudio
file_name = "audio.wav"
output_path = f"D:/File Storage/Documents/UTC/Thesis/WeatherMuse/audiocraft/weathermuse/output_files/{file_name}"
torchaudio.save(output_path, generated_audio[0].cpu(), model.sample_rate)
print(f"Audio saved to {output_path}")

######################################################################
######################################################################
"""Generate an excited voice sample via the Gradio API on port 8007."""
from gradio_client import Client
import shutil

client = Client("http://localhost:8007/")

print("=== Generating Excited Voice Sample ===")
result = client.predict(
    text="Oh my god, training is COMPLETE! This is absolutely INCREDIBLE! I can't believe how amazing this sounds! YES! We did it! This is the best day EVER!",
    lang_disp="English",
    spk_disp="Speaker Gared",
    instruct="Speak with extreme excitement, screaming with joy, high energy and bursting enthusiasm.",
    api_name="/run_instruct"
)

audio_path, status = result
print(f"Status: {status}")
print(f"Audio path: {audio_path}")

# Copy to project root for easy access
output = "voice_sample_excited.wav"
shutil.copy(audio_path, output)
print(f"\n🎉 SUCCESS: Saved to {output}")

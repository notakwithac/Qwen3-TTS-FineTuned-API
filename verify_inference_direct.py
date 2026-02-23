import logging
import torch
from inference_manager import InferenceManager
import soundfile as sf
import io

logging.basicConfig(level=logging.INFO)

def test_inference():
    print("=== Testing Direct Inference (No Mock) ===")
    
    try:
        # Load the manager
        manager = InferenceManager(use_flash_attn=True, idle_timeout_seconds=60)
        
        print("Attempting to generate voice design audio...")
        # Note: This will trigger our flash-attn fallback warning if not installed
        # but should succeed using eager attention and the correct numpy version.
        wav_bytes, sr = manager.generate_voice_design(
            text="The direct inference engine is now working correctly after the dependency fix.",
            instruct="A clear male voice.",
            language="English"
        )
        
        if wav_bytes and len(wav_bytes) > 0:
            print(f"SUCCESS: Generated {len(wav_bytes)} bytes of audio at {sr}Hz.")
            with open("final_verification.wav", "wb") as f:
                f.write(wav_bytes)
            print("Saved audio to final_verification.wav")
            return True
        else:
            print("FAILED: Generated empty audio.")
            return False
            
    except Exception as e:
        print(f"FAILED: Inference raised an exception: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_inference()

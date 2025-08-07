import os
import subprocess
import glob
import time
import re
import json
import random
from typing import List, Dict, Tuple, Optional
from tqdm import tqdm
import pysubs2
import ollama

# ---------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------
VIDEO_DIR      = "videos_to_process"
SRT_DIR        = "srt_outputs"
TEMP_AUDIO_DIR = "temp_audio"

WHISPER_MODEL  = "small"     # faster-whisper model size "small", "medium", "large"
DEVICE         = "cpu"        # "cpu" or "cuda"
COMPUTE_TYPE   = "int8"       # "int8" for CPU

# Ollama configuration
OLLAMA_MODEL = "llama3.1:8b"
OLLAMA_BASE_URL = "http://localhost:11434"

# Debug configuration
DEBUG_MODE = True  # Enable verbose Ollama debugging

# Load brand corrections database
def load_brand_corrections() -> Dict:
    """Load the brand corrections database from JSON file."""
    try:
        with open("brand_corrections.json", "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        print("Warning: brand_corrections.json not found. Using default brand list.")
        return {
            "brand_names": [
                "Acopa", "Adourne", "AvaMix", "Avantco Equipment", "Avantco Ice Machines",
                "Avantco Refrigeration", "AvaTek", "Backyard Pro", "Baker's Lane", "Boltic",
                "Carnival King", "CaterGator", "Certifications Guide", "Choice", "Clark Associates",
                "Clark Associates Charitable Foundation", "Cooking Performance Group",
                "Emperor's Select", "Estella Equipment", "Fryclone", "Garde",
                "Lancaster Table & Seating", "Lavex", "MainStreet Equipment", "Narvon Beverage",
                "New Roots", "Regency Equipment", "Schräf", "ServIt", "Vigor", "Waterloo",
                "WebstaurantStore", "Boltic Fans"
            ],
            "correction_examples": [],
            "common_misspellings": {}
        }

BRAND_DATA = load_brand_corrections()
BRAND_NAMES = BRAND_DATA["brand_names"]

# Create regex patterns for local corrections
_BRAND_REPLACEMENTS = {b.lower(): b for b in sorted(BRAND_NAMES, key=len, reverse=True)}
_BRAND_PATTERNS = [
    (re.compile(rf"\b{re.escape(low)}\b", flags=re.IGNORECASE), corr)
    for low, corr in _BRAND_REPLACEMENTS.items()
]

# ---------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------
def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def write_vtt_from_srt(srt_src: str | os.PathLike, vtt_path: str) -> None:
    pysubs2.load(srt_src, format="srt").save(vtt_path, format="vtt")

def extract_audio(video_path: str, wav_path: str) -> bool:
    # Try to find ffmpeg in common locations
    ffmpeg_paths = [
        "ffmpeg",  # System PATH
        r"C:\ffmpeg\bin\ffmpeg.exe",  # Common installation
        r"C:\Program Files\ffmpeg\bin\ffmpeg.exe",
        r"C:\Tools\ffmpeg\bin\ffmpeg.exe",
    ]
    
    ffmpeg_cmd = None
    for path in ffmpeg_paths:
        try:
            subprocess.run([path, "-version"], capture_output=True, check=False)
            ffmpeg_cmd = path
            break
        except (FileNotFoundError, OSError):
            continue
    
    if not ffmpeg_cmd:
        print("ERROR: ffmpeg not found! Please install ffmpeg and add it to PATH.")
        print("Download from: https://www.gyan.dev/ffmpeg/builds/")
        return False
    
    cmd = [
        ffmpeg_cmd, "-i", video_path,
        "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1", "-y", wav_path
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode:
        print(f"ffmpeg error for {video_path}: {res.stderr}")
    return res.returncode == 0

def _ts_srt(sec: float) -> str:
    ms = int(round((sec % 1) * 1000))
    s  = int(sec)
    m, s = divmod(s, 60)
    h, m = divmod(m, 60)
    return f"{h:02}:{m:02}:{s:02},{ms:03}"

def _ts_vtt(sec: float) -> str:
    ms = int(round((sec % 1) * 1000))
    s  = int(sec)
    m, s = divmod(s, 60)
    h, m = divmod(m, 60)
    return f"{h:02}:{m:02}:{s:02}.{ms:03}"

def correct_brand_names_local(segments) -> int:
    """Local brand correction using regex patterns."""
    fixes = 0
    for seg in segments:
        txt = seg.text
        for pattern, repl in _BRAND_PATTERNS:
            txt, n = pattern.subn(repl, txt)
            fixes += n
        seg.text = txt
    return fixes

def write_subtitles(segments, srt_path: str, vtt_path: str) -> None:
    # SRT
    with open(srt_path, "w", encoding="utf-8") as f:
        for i, seg in enumerate(segments, 1):
            f.write(f"{i}\n")
            f.write(f"{_ts_srt(seg.start)} --> {_ts_srt(seg.end)}\n")
            f.write(seg.text.strip() + "\n\n")
    # VTT
    with open(vtt_path, "w", encoding="utf-8") as f:
        f.write("WEBVTT\n\n")  # Add VTT header
        for seg in segments:
            f.write(f"{_ts_vtt(seg.start)} --> {_ts_vtt(seg.end)}\n")
            f.write(seg.text.strip() + "\n\n")

def transcribe_audio_faster_whisper(wav_path: str):
    from faster_whisper import WhisperModel
    model = WhisperModel(WHISPER_MODEL, device=DEVICE, compute_type=COMPUTE_TYPE)
    seg_gen, info = model.transcribe(wav_path, beam_size=5, language=None, vad_filter=True)
    print(f"Detected language '{info.language}' (p={info.language_probability:.2f})")
    return list(seg_gen)

# ---------------------------------------------------------------
# Ollama NER System
# ---------------------------------------------------------------
class OllamaNER:
    def __init__(self, model: str = OLLAMA_MODEL, base_url: str = OLLAMA_BASE_URL):
        self.model = model
        self.base_url = base_url
        self.client = ollama.Client(host=base_url)
        self.brand_data = BRAND_DATA
        
        # Test Ollama connection on startup
        self.test_ollama_connection()
        
    def test_ollama_connection(self) -> None:
        """Test Ollama connection with a simple hello message."""
        if DEBUG_MODE:
            print("\n" + "="*50)
            print("CONNECT: TESTING OLLAMA CONNECTION")
            print("="*50)
            
            try:
                test_response = self.client.chat(
                    model=self.model,
                    messages=[{
                        "role": "user", 
                        "content": "Hello! Please respond with just 'Hello, I am working!' to test the connection."
                    }],
                    options={"temperature": 0.1}
                )
                
                print(f"SUCCESS: Ollama Connection Test:")
                print(f"   Model: {self.model}")
                print(f"   Response: {test_response['message']['content'].strip()}")
                print("="*50 + "\n")
                
            except Exception as e:
                print(f"ERROR: Ollama Connection Test FAILED: {e}")
                print("="*50 + "\n")
    
    def check_ollama_health(self) -> bool:
        """Check if Ollama is running and the model is available."""
        try:
            models = self.client.list()
            available_models = [model.model for model in models.models]
            if self.model not in available_models:
                print(f"Warning: Model {self.model} not found. Available models: {available_models}")
                return False
            return True
        except Exception as e:
            print(f"Ollama health check failed: {e}")
            return False
    
    def select_relevant_examples(self, text: str, num_examples: int = 3) -> List[Dict]:
        """Select relevant correction examples based on text content."""
        examples = self.brand_data.get("correction_examples", [])
        if not examples:
            return []
        
        # Simple relevance scoring based on brand mentions
        text_lower = text.lower()
        scored_examples = []
        
        for example in examples:
            score = 0
            for brand in example.get("brands_corrected", []):
                if brand.lower() in text_lower:
                    score += 2
                # Check for partial matches or common misspellings
                for misspelling in self.brand_data.get("common_misspellings", {}).get(brand.lower(), []):
                    if misspelling.lower() in text_lower:
                        score += 1
            scored_examples.append((score, example))
        
        # Sort by score and return top examples
        scored_examples.sort(key=lambda x: x[0], reverse=True)
        selected = [ex[1] for ex in scored_examples[:num_examples]]
        
        # If no high-scoring examples, add some random ones for few-shot learning
        if len(selected) < num_examples:
            remaining = [ex[1] for ex in scored_examples[num_examples:]]
            random.shuffle(remaining)
            selected.extend(remaining[:num_examples - len(selected)])
        
        return selected
    
    def create_correction_prompt(self, transcript: str) -> str:
        """Create an enhanced prompt with examples for brand name correction."""
        brand_list = "\n".join(f"- {brand}" for brand in self.brand_data["brand_names"])
        
        # Select relevant examples
        examples = self.select_relevant_examples(transcript, num_examples=3)
        
        examples_text = ""
        if examples:
            examples_text = "\nHere are some examples of correct brand name usage:\n\n"
            for i, example in enumerate(examples, 1):
                examples_text += f"Example {i}:\n"
                examples_text += f"Incorrect: \"{example['incorrect']}\"\n"
                examples_text += f"Correct: \"{example['correct']}\"\n\n"
        
        prompt = f"""You are an expert text editor specializing in brand name correction for subtitle files. Your task is to correct any misspellings, variations, or incorrect capitalizations of these specific brand names:

{brand_list}

IMPORTANT RULES:
1. Only correct the brand names listed above - do not change any other words
2. Preserve the exact capitalization and spacing shown in the brand list
3. Keep all subtitle timing codes and formatting exactly as provided
4. Do not add, remove, or modify any other content{examples_text}

Subtitle content to correct:
{transcript}

Please return the corrected subtitle file with only the brand names fixed:"""
        
        return prompt
    
    def correct_brands_with_ollama(self, srt_path: str, max_retries: int = 2) -> Optional[str]:
        """Use Ollama to correct brand names in the SRT file."""
        try:
            with open(srt_path, "r", encoding="utf-8") as f:
                content = f.read()
        except Exception as e:
            print(f"Could not read {srt_path}: {e}")
            return None
        
        if not self.check_ollama_health():
            print("Ollama health check failed. Skipping AI correction.")
            return None
        
        prompt = self.create_correction_prompt(content)
        
        if DEBUG_MODE:
            print("\n" + "="*60)
            print("AI: OLLAMA BRAND CORRECTION DEBUG")
            print("="*60)
            print(f"INPUT: INPUT SRT CONTENT (first 200 chars):")
            print(f"   {content[:200]}...")
            print(f"\nPROMPT: PROMPT SENT TO OLLAMA (first 300 chars):")
            print(f"   {prompt[:300]}...")
            print("="*60)
        
        for attempt in range(max_retries + 1):
            try:
                print(f"Attempting Ollama correction (attempt {attempt + 1}/{max_retries + 1})...")
                
                if DEBUG_MODE:
                    print(f"\nPROCESS: Sending request to Ollama (attempt {attempt + 1}/{max_retries + 1})...")
                    
                response = self.client.chat(
                    model=self.model,
                    messages=[
                        {
                            "role": "system", 
                            "content": "You are a professional subtitle editor. Correct only the specified brand names while preserving all formatting."
                        },
                        {
                            "role": "user", 
                            "content": prompt
                        }
                    ],
                    options={
                        "temperature": 0.1,  # Low temperature for consistent corrections
                        "top_p": 0.9,
                        "num_predict": 2048  # Limit response length
                    }
                )
                
                corrected_content = response['message']['content'].strip()
                
                if DEBUG_MODE:
                    print(f"\nSUCCESS: OLLAMA RESPONSE (first 300 chars):")
                    print(f"   {corrected_content[:300]}...")
                    print(f"\nDEBUG: CHANGES DETECTED:")
                    
                    # Simple change detection
                    original_lines = content.split('\n')
                    corrected_lines = corrected_content.split('\n')
                    changes = 0
                    for i, (orig, corr) in enumerate(zip(original_lines, corrected_lines)):
                        if orig != corr and '-->' not in orig:  # Skip timestamp lines
                            print(f"   Line {i+1}: '{orig.strip()}' → '{corr.strip()}'")
                            changes += 1
                    if changes == 0:
                        print(f"   No brand corrections detected")
                    else:
                        print(f"   Total lines changed: {changes}")
                    print("="*60 + "\n")
                
                # Basic validation - check if response looks like a subtitle file
                if "-->" in corrected_content and len(corrected_content) > len(content) * 0.5:
                    print("Ollama correction completed successfully.")
                    return corrected_content
                else:
                    print(f"Ollama response doesn't look like a valid subtitle file (attempt {attempt + 1})")
                    if attempt < max_retries:
                        time.sleep(2)  # Brief pause before retry
                    
            except Exception as e:
                print(f"Ollama error (attempt {attempt + 1}): {e}")
                if attempt < max_retries:
                    time.sleep(3)  # Longer pause before retry
        
        print(f"All Ollama correction attempts failed. Using original content.")
        return None

# ---------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------
def main() -> None:
    # prerequisite check
    try:
        import faster_whisper  # noqa: F401
    except ImportError:
        print("Missing dependency. Run: pip install faster-whisper")
        return

    for d in (VIDEO_DIR, SRT_DIR, TEMP_AUDIO_DIR):
        ensure_dir(d)

    print("="*60)
    print("VIDEO: TRANSCRIPTION WITH OLLAMA DEBUG MODE")
    print("="*60)
    
    # Initialize Ollama NER system (will test connection automatically)
    ner_system = OllamaNER()
    print(f"Using Ollama model: {OLLAMA_MODEL}")

    videos = []
    for ext in ("*.mp4", "*.mov", "*.mkv", "*.avi", "*.webm", "*.wmv", "*.flv"):
        videos.extend(glob.glob(os.path.join(VIDEO_DIR, ext)))

    if not videos:
        print(f"No videos found in {VIDEO_DIR}")
        return

    for vid in tqdm(videos, desc="Videos"):
        base = os.path.splitext(os.path.basename(vid))[0]
        wav = os.path.join(TEMP_AUDIO_DIR, base + ".wav")
        srt = os.path.join(SRT_DIR, base + ".srt")
        vtt = os.path.join(SRT_DIR, base + ".vtt")

        # skip if fully processed
        if os.path.exists(srt) and os.path.exists(vtt):
            print(f"{base}: subtitles already exist – skipping")
            continue

        # 1) extract audio
        if not os.path.exists(wav):
            print(f"Extracting audio  {wav}")
            if not extract_audio(vid, wav):
                continue
        else:
            print(f"Using cached audio {wav}")

        # 2) transcribe + local fixes (only if SRT missing)
        if not os.path.exists(srt):
            print(f"Transcribing {base}")
            segs = transcribe_audio_faster_whisper(wav)
            local_fixes = correct_brand_names_local(segs)
            print(f"Local brand fixes applied: {local_fixes}")
            write_subtitles(segs, srt, vtt)
        else:
            print(f"{base}: found existing SRT – skipping transcription")

        # 3) Ollama NER correction (always attempt)
        print(f"{base}: running Ollama brand correction")
        corrected_content = ner_system.correct_brands_with_ollama(srt)
        
        if corrected_content:
            # Save corrected content
            with open(srt, "w", encoding="utf-8") as f:
                f.write(corrected_content)
            # Regenerate VTT from corrected SRT
            write_vtt_from_srt(srt, vtt)
            print(f"{base}: Ollama corrections applied to SRT + VTT")
        else:
            print(f"{base}: Ollama correction failed – keeping original subtitles")

        # 4) cleanup
        try:
            os.remove(wav)
        except OSError:
            pass
        print("-" * 40)

    print("All videos processed")

# ---------------------------------------------------------------
if __name__ == "__main__":
    main()
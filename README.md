# Automatic Video Transcription - Ollama Enhanced Version

This project provides an automated workflow for transcribing video files into subtitle formats (SRT and VTT). It utilizes a "hot folder" to monitor for new video files, processes them using `faster-whisper` for transcription, and then uses **Ollama with Llama 3.1** for intelligent brand name correction in the generated subtitles.

## Key Improvements in This Version

- **Local AI Processing**: Uses Ollama with Llama 3.1 instead of OpenAI for brand correction (no API costs!)
- **Enhanced NER System**: Intelligent brand name correction with stored examples and few-shot learning
- **Better VTT Support**: Added proper WEBVTT header for VTT files
- **Improved Security**: No API keys required - everything runs locally
- **Smart Example Selection**: Automatically selects relevant correction examples based on content
- **Robust Error Handling**: Multiple retry attempts and fallback mechanisms

## How It Works

1.  **File-Based Trigger**: A user drops a video file into the `INBOX` directory.
2.  **Folder Monitoring**: The `watch_folder.py` script, using the `watchdog` library, detects the new file.
3.  **Processing Pipeline**:
    *   The video is moved to the `videos_to_process` directory.
    *   The `transcribe_videos_ollama.py` script is triggered.
    *   **Audio Extraction**: `ffmpeg` extracts the audio from the video into a temporary `.wav` file, stored in the `temp_audio` directory.
    *   **Transcription**: The audio is transcribed using the `faster-whisper` library. The resulting segments are saved to an SRT file in the `srt_outputs` directory.
    *   **Brand Name Correction**: The generated SRT file is processed by Ollama (Llama 3.1) to intelligently correct any misspellings of brand names using stored examples and few-shot learning.
    *   **VTT Generation**: A VTT version of the corrected SRT file is created.
4.  **Finalization**: The original video and the generated SRT and VTT files are moved to the `FINAL_OUTPUT` directory. The temporary audio file is deleted.

## Directory Structure

*   `INBOX/`: The "hot folder." Place video files here to begin processing.
*   `videos_to_process/`: A temporary holding area for videos currently being processed.
*   `temp_audio/`: Stores temporary WAV audio files extracted from the videos.
*   `srt_outputs/`: A temporary holding area for the generated SRT and VTT files before they are moved to the final output directory.
*   `FINAL_OUTPUT/`: The destination for the original video and its final SRT and VTT subtitle files.
*   `.venv/`: Contains the Python virtual environment and dependencies.

## Scripts

*   `watch_folder.py`: Monitors the `INBOX` directory for new video files and orchestrates the transcription process.
*   `transcribe_videos_ollama.py`: The core script that handles audio extraction, transcription, and Ollama-based brand name correction.
*   `brand_corrections.json`: Database of brand names, correction examples, and common misspellings for enhanced NER.

## Setup and Usage

### Prerequisites
*   Python 3.8 or higher
*   `ffmpeg` must be installed and available in the system's PATH
*   **Ollama installed and running** with Llama 3.1 model

### Installation

1.  **Install and Setup Ollama**:
    *   Download and install Ollama from [ollama.ai](https://ollama.ai)
    *   Pull the Llama 3.1 model (choose based on your system):
        ```bash
        # For most systems (8GB+ RAM recommended)
        ollama pull llama3.1:8b
        
        # For systems with limited RAM (4GB+ RAM)
        ollama pull llama3.1:7b
        
        # For high-performance systems (16GB+ RAM)
        ollama pull llama3.1:70b
        ```
    *   Start Ollama server (if not running):
        ```bash
        ollama serve
        ```
    *   **Note**: The default configuration uses `llama3.1:8b`. To use a different model, edit the `OLLAMA_MODEL` variable in `transcribe_videos_ollama.py`.

2.  Clone this repository or copy the project files.

3.  Create a Python virtual environment:
    ```bash
    python -m venv .venv
    ```

4.  Activate the virtual environment:
    *   **Windows**:
        ```bash
        .venv\Scripts\activate
        ```
    *   **macOS/Linux**:
        ```bash
        source .venv/bin/activate
        ```

5.  Install the required Python packages:
    ```bash
    pip install -r requirements.txt
    ```

### Configuration

1.  **Test Ollama Setup**:
    Test if Ollama is working by running:
    ```bash
    ollama list
    ```
    You should see `llama3.1:8b` in the list of installed models.

2.  **Optional Model Configuration**:
    *   Open `transcribe_videos_ollama.py` and adjust the `WHISPER_MODEL` size (`small`, `medium`, `large`) depending on your hardware and desired accuracy.
    *   Change `DEVICE` to `"cuda"` if you have a compatible GPU.
    *   Modify `OLLAMA_MODEL` if you want to use a different Llama model variant.

3.  **Customize Brand Corrections**:
    *   Edit `brand_corrections.json` to add or modify brand names, examples, and common misspellings.
    *   The system uses these examples for few-shot learning to improve correction accuracy.

### Running the Application

1.  Ensure your virtual environment is activated and Ollama is running.

2.  Start the folder monitoring script:
    ```bash
    python watch_folder.py
    ```

3.  The script will now be watching the `INBOX` folder. To start a transcription, simply copy or move a video file into this folder.

4.  Monitor the console output for progress updates and any errors.

5.  Once processing is complete, find your video and subtitle files in the `FINAL_OUTPUT` directory.

## Supported Video Formats

The system supports the following video formats:
- MP4
- MOV
- MKV
- AVI
- WebM
- WMV
- FLV

## Customizing Brand Names

To modify the brand correction system:
1. **Add/Edit Brand Names**: Update the `brand_names` array in `brand_corrections.json`
2. **Add Examples**: Include correction examples in the `correction_examples` array for better accuracy
3. **Add Misspellings**: Update the `common_misspellings` object with known variations

The system automatically selects relevant examples for each correction task using intelligent matching.

## Troubleshooting

### ffmpeg not found
Make sure ffmpeg is installed and accessible from your command line. You can test this by running:
```bash
ffmpeg -version
```

### Ollama errors
- **Service not running**: Start Ollama with `ollama serve`
- **Model not found**: Install the model with `ollama pull llama3.1:8b`
- **Connection refused**: Check if Ollama is running on port 11434 with `ollama list`
- **Out of memory**: Try switching to a smaller model (`llama3.1:7b`) in `transcribe_videos_ollama.py`
- **Slow responses**: Consider using GPU acceleration if available, or switch to a smaller model
- **Permission issues**: Try running Ollama as administrator (Windows) or with sudo (Linux/Mac)

### Ollama Performance Tips
- **GPU Acceleration**: Ollama automatically uses GPU if available (NVIDIA CUDA, Apple Metal, AMD ROCm)
- **Memory Usage**: Each model has different RAM requirements:
  - `7b` models: ~4GB RAM
  - `8b` models: ~8GB RAM  
  - `70b` models: ~40GB RAM
- **Response Speed**: Smaller models respond faster but may be less accurate for complex corrections
- **First Run**: The first correction will be slower as Ollama loads the model into memory

### File processing stuck
- Check the console output for error messages
- Ensure the video file has finished copying to the INBOX before the script detects it
- Try restarting the watch_folder.py script

## Dependencies

This project relies on the following main packages:
*   `faster-whisper`: For efficient audio transcription
*   `ollama`: For local AI model interaction
*   `watchdog`: For monitoring the file system
*   `pysubs2`: For working with subtitle formats
*   `tqdm`: For displaying progress bars
*   `torch`, `torchaudio`, `torchvision`: Required by faster-whisper

See `requirements.txt` for the complete list of dependencies.

## Advantages of This Ollama-Based Approach

- **No API Costs**: Everything runs locally, no per-request charges
- **Privacy**: No data sent to external services
- **Reliability**: No internet dependency or rate limits
- **Customizable**: Full control over the AI model and correction logic
- **Fast**: Once loaded, Ollama responses are very quick
- **Learning**: Few-shot learning with examples improves accuracy over time
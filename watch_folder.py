import os
import shutil
import subprocess
import time
import logging
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

# --- Configuration ---
# Assuming watch_folder.py is in "SRT from Video Test Improved" directory
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

FILESHARE = os.path.join(r"\\therestaurantstore.com", "video", "Video", "AI-Transcribing")
INBOX_DIR = os.path.join(FILESHARE, "INBOX")
VIDEOS_TO_PROCESS_DIR = os.path.join(FILESHARE, "videos_to_process")
SRT_OUTPUTS_DIR = os.path.join(FILESHARE, "srt_outputs")
FINAL_OUTPUT_DIR = os.path.join(FILESHARE, "FINAL_OUTPUT")
TEMP_AUDIO_DIR = os.path.join(FILESHARE, "temp_audio")

VENV_PYTHON_EXECUTABLE = os.path.join(PROJECT_ROOT, ".venv", "Scripts", "python.exe")
TRANSCRIPTION_SCRIPT_PATH = os.path.join(PROJECT_ROOT, "transcribe_videos_ollama.py")

VIDEO_EXTENSIONS = ('.mp4', '.mov', '.mkv', '.avi', '.webm', '.wmv', '.flv')

# Setup logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
		    filename= os.path.join(FILESHARE, 'transcription-log.txt'))

class VideoHandler(FileSystemEventHandler):
    def wait_for_file_stability(self, file_path, stability_checks=3, check_interval=3):
        """
        Waits for a file to stabilize by checking its size.
        Args:
            file_path (str): The path to the file.
            stability_checks (int): Number of times the size must be consistent.
            check_interval (int): Seconds between size checks.
        Returns:
            bool: True if the file stabilized, False otherwise.
        """
        file_name = os.path.basename(file_path)
        logging.info(f"Checking stability for {file_name}...")
        last_size = -1
        stable_count = 0
        max_attempts = stability_checks + 5 # Allow a few extra attempts for OSError or initial fluctuations

        attempts = 0
        while stable_count < stability_checks and attempts < max_attempts:
            attempts += 1
            try:
                if not os.path.exists(file_path):
                    logging.warning(f"File {file_name} disappeared during stability check.")
                    return False
                
                current_size = os.path.getsize(file_path)
                logging.debug(f"File: {file_name}, Attempt: {attempts}, Current size: {current_size}, Last size: {last_size}, Stable count: {stable_count}")

            except OSError as e:
                logging.warning(f"OSError while checking {file_name}: {e}. Retrying in {check_interval}s...")
                time.sleep(check_interval)
                continue # Retry the check

            if current_size == last_size and current_size != 0: # Ensure file is not empty and stable
                stable_count += 1
            else:
                stable_count = 0 # Reset if size changes or is zero initially
            
            last_size = current_size
            
            if stable_count < stability_checks:
                time.sleep(check_interval)
        
        if stable_count >= stability_checks:
            logging.info(f"File {file_name} is stable with size {last_size}.")
            return True
        else:
            logging.warning(f"File {file_name} did not stabilize after {attempts} attempts. Last size: {last_size}. Stable count: {stable_count}.")
            return False

    def on_created(self, event):
        if event.is_directory:
            return

        src_path = event.src_path
        file_name = os.path.basename(src_path)

        # Check for video extensions first
        if not file_name.lower().endswith(VIDEO_EXTENSIONS):
            # This check can be uncommented if you want to log all ignored files
            # logging.debug(f"Ignoring non-video file on creation: {file_name}")
            return

        logging.info(f"New video candidate detected: {file_name} at {src_path}")

        # Wait for the file to be stable
        if not self.wait_for_file_stability(src_path):
            logging.warning(f"File {file_name} did not stabilize or is empty. Skipping.")
            return
        
        # Ensure directories exist 
        for dir_path in [VIDEOS_TO_PROCESS_DIR, SRT_OUTPUTS_DIR, FINAL_OUTPUT_DIR]:
            os.makedirs(dir_path, exist_ok=True)

        # 1. Move file from INBOX to videos_to_process
        processing_video_path = os.path.join(VIDEOS_TO_PROCESS_DIR, file_name)
        try:
            logging.info(f"Moving '{file_name}' to '{VIDEOS_TO_PROCESS_DIR}' for processing.")
            shutil.move(src_path, processing_video_path)
            logging.info(f"Successfully moved '{file_name}' to processing directory.")
        except Exception as e:
            # If src_path doesn't exist here, it might be a quick duplicate event for a file already processed
            if not os.path.exists(src_path):
                 logging.warning(f"Source file {src_path} no longer exists. Possibly a duplicate event or already processed.")
                 return
            logging.error(f"Error moving file {file_name} to processing directory: {e}")
            return

        # 2. Run your transcription script using the venv python
        logging.info(f"Starting transcription process for '{file_name}'...")
        try:
            process = subprocess.Popen(
                [VENV_PYTHON_EXECUTABLE, TRANSCRIPTION_SCRIPT_PATH],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                cwd=PROJECT_ROOT,
                encoding='utf-8',
                errors='replace'  # Handle encoding errors gracefully
            )
            stdout, stderr = process.communicate()

            if process.returncode == 0:
                logging.info(f"Transcription script completed successfully for '{file_name}'.")
                if stdout: logging.info(f"Script output:\n{stdout}")
            else:
                logging.error(f"Transcription script failed for '{file_name}' with return code {process.returncode}.")
                if stderr: logging.error(f"Script error output:\n{stderr}")

        except Exception as e:
            logging.error(f"An exception occurred while running the transcription script: {e}")
        
        # 3. Move processed files to FINAL_OUTPUT
        base_name, _ = os.path.splitext(file_name)
        final_video_path = os.path.join(FINAL_OUTPUT_DIR, file_name)
        srt_file_name = base_name + ".srt"
        vtt_file_name = base_name + ".vtt"
        # Original video is now at processing_video_path
        # SRT/VTT outputs are expected in SRT_OUTPUTS_DIR

        src_srt_path = os.path.join(SRT_OUTPUTS_DIR, srt_file_name)
        src_vtt_path = os.path.join(SRT_OUTPUTS_DIR, vtt_file_name)
        final_srt_path = os.path.join(FINAL_OUTPUT_DIR, srt_file_name)
        final_vtt_path = os.path.join(FINAL_OUTPUT_DIR, vtt_file_name)

        try:
            # Move original video from videos_to_process to FINAL_OUTPUT
            if os.path.exists(processing_video_path):
                 shutil.move(processing_video_path, final_video_path)
                 logging.info(f"Moved '{file_name}' to '{FINAL_OUTPUT_DIR}'.")
            else:
                logging.warning(f"Processed video '{processing_video_path}' not found for moving to final output. It might have been moved by the script or an error occurred.")

            # Move SRT file
            if os.path.exists(src_srt_path):
                shutil.move(src_srt_path, final_srt_path)
                logging.info(f"Moved '{srt_file_name}' to '{FINAL_OUTPUT_DIR}'.")
            else:
                logging.warning(f"SRT file '{src_srt_path}' not found. Skipping move.")

            # Move VTT file
            if os.path.exists(src_vtt_path):
                shutil.move(src_vtt_path, final_vtt_path)
                logging.info(f"Moved '{vtt_file_name}' to '{FINAL_OUTPUT_DIR}'.")
            else:
                logging.warning(f"VTT file '{src_vtt_path}' not found. Skipping move.")

        except Exception as e:
            logging.error(f"Error moving files to FINAL_OUTPUT for '{base_name}': {e}")

        logging.info(f"Processing finished for '{file_name}'.")
        logging.info("-" * 40)


if __name__ == "__main__":
    for dir_path in [INBOX_DIR, VIDEOS_TO_PROCESS_DIR, SRT_OUTPUTS_DIR, FINAL_OUTPUT_DIR, TEMP_AUDIO_DIR]:
        os.makedirs(dir_path, exist_ok=True)

    logging.info(f"Starting hot folder monitor on: {INBOX_DIR}")
    event_handler = VideoHandler()
    observer = Observer()
    observer.schedule(event_handler, INBOX_DIR, recursive=False)
    observer.start()

    try:
        while True:
            time.sleep(5)
    except KeyboardInterrupt:
        logging.info("Stopping hot folder monitor.")
        observer.stop()
    except Exception as e:
        logging.error(f"An unexpected error occurred in the observer: {e}")
        observer.stop()
    observer.join()
    logging.info("Hot folder monitor shut down.")
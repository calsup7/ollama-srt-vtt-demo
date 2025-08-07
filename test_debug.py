#!/usr/bin/env python3
"""
Quick test script to debug Ollama brand corrections.
This will test the debug version without processing videos.
"""

import os
import shutil
from transcribe_videos_ollama_debug import OllamaNER

def test_ollama_on_existing_srt():
    """Test Ollama correction on an existing SRT file."""
    
    # Check if we have any SRT files to test with
    test_files = []
    
    # Look for SRT files in FINAL_OUTPUT
    final_output_dir = "FINAL_OUTPUT"
    if os.path.exists(final_output_dir):
        for file in os.listdir(final_output_dir):
            if file.endswith('.srt'):
                test_files.append(os.path.join(final_output_dir, file))
    
    # Look for SRT files in legacy
    legacy_dir = "legacy"
    if os.path.exists(legacy_dir):
        for file in os.listdir(legacy_dir):
            if file.endswith('.srt'):
                test_files.append(os.path.join(legacy_dir, file))
    
    if not test_files:
        print("ERROR: No SRT files found to test with.")
        return
    
    # Use the first SRT file we find
    test_srt = test_files[0]
    print(f"Testing Ollama correction on: {test_srt}")
    print("="*60)
    
    # Initialize the NER system (will show connection test)
    ner_system = OllamaNER()
    
    # Test the correction
    corrected_content = ner_system.correct_brands_with_ollama(test_srt)
    
    if corrected_content:
        print("SUCCESS: Ollama correction completed successfully!")
        
        # Save the corrected version for comparison
        test_output = "test_corrected_output.srt"
        with open(test_output, "w", encoding="utf-8") as f:
            f.write(corrected_content)
        print(f"Corrected version saved as: {test_output}")
        print("Compare the original and corrected files to see changes.")
        
    else:
        print("ERROR: Ollama correction failed.")

if __name__ == "__main__":
    print("OLLAMA DEBUG TEST")
    print("="*60)
    test_ollama_on_existing_srt()
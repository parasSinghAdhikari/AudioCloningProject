# AudioCloningProject
## Audio Cloning with Unsloth and CSM-1B

This project demonstrates how to create a voice cloning system using the Unsloth framework and CSM-1B model. The pipeline downloads audio from YouTube, processes it, creates a dataset, and fine-tunes a model to clone voices.

## Overview

The notebook performs the following steps:
1. **Audio Download**: Downloads audio from YouTube videos
2. **Audio Processing**: Cleans and normalizes audio files
3. **Transcription**: Uses OpenAI Whisper to transcribe audio with word timestamps
4. **Dataset Creation**: Creates chunks of audio-text pairs for training
5. **Model Fine-tuning**: Fine-tunes CSM-1B model for voice cloning
6. **Voice Generation**: Generates speech in the cloned voice

## Requirements

```python
!pip install openai-whisper
!pip install yt_dlp
!pip install unsloth[all]
```

Required libraries:
- `whisper` - For speech transcription
- `yt_dlp` - For YouTube video downloading
- `pydub` - For audio processing
- `unsloth` - For efficient model fine-tuning
- `transformers` - For model handling
- `datasets` - For dataset management
- `pandas`, `numpy` - For data manipulation

## Project Structure

```
/content/
├── audio/                    # Downloaded raw audio files
├── clean_audio/             # Processed and cleaned audio
├── transcripts/             # JSON transcription files
├── our_dataset/            # Final dataset
│   └── dataset_json.json   # Training dataset
├── myaudio_segments/       # Audio chunks (20-30 seconds each)
└── outputs/                # Model training outputs
```

## Step-by-Step Process

### 1. Audio Download
```python
# Downloads audio from YouTube using yt_dlp
url = ['https://www.youtube.com/watch?v=mKBbP4T5fbk&t=142s&ab_channel=SpeakEnglishWithTiffani']
```

### 2. Audio Cleaning
Applies FFmpeg filters to improve audio quality:
- High-pass filter (removes noise below 90Hz)
- FFT-based denoiser
- Loudness normalization
- Dynamic audio normalization
- Converts to 32kHz mono

### 3. Transcription
Uses OpenAI Whisper Large model to:
- Generate word-level timestamps
- Create accurate transcriptions
- Save results as JSON files

### 4. Dataset Creation
Creates training dataset with:
- **Audio chunks**: 20-30 second segments
- **Aligned text**: Corresponding transcribed text
- **Speaker ID**: Source identifier (0 for single speaker)

Dataset format:
```json
{
  "audio": {"path": "path/to/audio/chunk.wav"},
  "text": "Transcribed text for this audio segment",
  "source": "0"
}
```

### 5. Model Fine-tuning
- Uses Unsloth's CSM-1B model
- Applies LoRA (Low-Rank Adaptation) for efficient training
- Configures training parameters for voice cloning

### 6. Voice Generation
Generates speech using the fine-tuned model:
- Takes text input and speaker reference
- Produces audio in the cloned voice
- Saves output as WAV files

## Key Features

### Audio Processing Pipeline
- Automatic audio quality enhancement
- Noise reduction and normalization
- Consistent sample rate conversion

### Smart Chunking
- Creates 20-30 second audio segments
- Respects sentence boundaries
- Maintains audio-text alignment

### Efficient Training
- Uses LoRA for memory-efficient fine-tuning
- 4-bit quantization support
- Gradient checkpointing

## Usage

### Basic Usage
1. **Set your YouTube URL** in the download section
2. **Run all cells** sequentially
3. **Wait for processing** (download → clean → transcribe → train)
4. **Generate cloned speech** using the final cells

### Custom Text Generation
```python
text = "Your custom text here"
speaker_id = 0
# Generate audio using the trained model
```

### Voice Cloning
```python
# Use reference audio and generate new speech
cloned = rawTestDs[1]["audio"]["array"]
cloned_text = rawTestDs[1]["text"]
# Generate speech in the cloned voice
```

## Training Configuration

- **Model**: CSM-1B with LoRA adaptation
- **Batch Size**: 1 with gradient accumulation (4 steps)
- **Learning Rate**: 2e-4
- **Epochs**: 3
- **Audio Sample Rate**: 24kHz
- **Max Audio Length**: Variable (based on dataset)

## Output Files

- `finetuned.wav` - Basic text-to-speech output
- `clonedVoice.wav` - Voice-cloned speech output
- Model checkpoints in `/content/outputs/`
- Training logs in `/content/logs/`

## Performance Tips

1. **GPU Required**: Use CUDA-enabled environment
2. **Memory**: Ensure sufficient VRAM (8GB+ recommended)
3. **Audio Quality**: Higher quality input = better cloning results
4. **Training Time**: 3 epochs typically take 30-60 minutes

## Troubleshooting

### Common Issues
- **CUDA Out of Memory**: Reduce batch size or max sequence length
- **Audio Loading Errors**: Check file formats and paths
- **Poor Voice Quality**: Increase training epochs or improve input audio

### File Path Issues
Ensure all paths are correctly set:
```python
json_file = "/content/transcripts/your_file.json"
audio_file = "/content/clean_audio/your_file.wav"
```

## Results

The system can:
- Clone voices from just a few minutes of audio
- Generate natural-sounding speech
- Maintain speaker characteristics (tone, style, accent)
- Handle various text inputs

## Limitations

- Requires good quality source audio
- Performance depends on training data quantity
- May not capture all nuanced speech patterns
- Computational requirements are significant

## License

This project uses various open-source libraries. Please check individual library licenses for commercial use.

## Acknowledgments

- OpenAI Whisper for transcription
- Unsloth for efficient fine-tuning
- CSM-1B model developers
- YouTube content creators for audio samples

---

**Note**: This is for educational and research purposes. Always ensure you have proper permissions before cloning someone's voice.

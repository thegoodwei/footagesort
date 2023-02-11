Footage_Sort - An AI-Powered Video Editing Tool
UNSTABLE BUILD -- This project is in early beta, use at your own risk, or feel free to contribute.

Use AI to Edit Highlights of Long Videos
AI Soundbite Extractor, with linked video

Footagesort is a project that uses artificial intelligence and natural language processing to extract relevant quotes from long videos and interviews. Given a an audio file, it transcribes in timecode for an .srt transcript file, combining subtitles to complete ideas and comparing OpenAI API long summaries with embeddings to find the most relevant quotes.
The output is an edited .srt file with the desired quantity of relevant soundbites and a video timeline .edl file that can be edited in any NLE software.


The application transcribes audio into complete ideas and subtitles in .srt format.
OpenAi provides a relevancy search by embeddings similiarity score, based on a user-defined prompt and autokeywords from script summarization.

The user can set the target clip_length and the number of clips to return a media file, and the program will use the SRT and EDLkit to create an edited .EDL file with the edit decisions. This file can then be used in ffmpeg to render an AI-edited highlight video summary directly in video format. 


Libraries Used
Python3.8
SRT - To parse and manipulate the subtitles.
EDLkit - To create the edit decision list (EDL) for the edited video.
Whisperx - To transcribe audio files into .srt subtitles.
Numpy - For numerical computation and data manipulation.
MoviePy - To render the output media with ffmpeg

Installing
Clone the repository to your local machine:

```bash
git clone https://github.com/thegoodwei/footagesort.git
```

Install the required packages:

```bash
pip install -r requirements.txt

Functionality
transcribe_audio - Transcribes audio files into .srt subtitles using Whisper.
extract_soundbites - Searches for relevant clips based on the user-defined prompt and extract the desired number of soundbites.
generate_edl - Uses the extracted soundbites and EDLkit to create an edited .EDL file with the edit decisions.
render_video - Uses FFmpeg to render the AI-edited highlight video summary.

Usage
Run the program and you will be prompted for the following:
- Enter the file path for the audio file (support .mp3, .m4a, or .mp4).
- Enter the desired length for each soundbite.
- Enter the desired number of soundbites to extract.
- Enter the keywords to prompt for relevancy ranking.

Currently command-line only.

bash
python main.py --input_file [INPUT_AUDIO/VIDEO_FILE] --clip_length [CLIP_LENGTH] --num_clips [NUMBER_OF_CLIPS] --prompt [PROMPT] 


Contributing

Feel free to submit pull requests or issues. Any contribution is appreciated

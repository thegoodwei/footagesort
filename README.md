# footagesort
##Use openai to edit highlights of long videos and interviews
###a Subtitle Soundbite Extractor

This project takes an .srt video transcript file, concatenates subtitles to the desired search length, and uses OpenAI API to find the most relevant quotes to the user prompt. The application outputs an edited .srt of the user-prompted quantity of relevant soundbites and creates a video timeline .edl file with a placeholder fullvideo for the user to edit with in any NLE software.

This program is used to extract relevant subtitles from a given transcript and store them in a database along with their corresponding embeddings. The user is prompted for various parameters such as the title of the project, transcript file, duration of each block of text, number of soundbites to be shown, and keywords for relevancy ranking.

Getting Started
Libraries Used
    Python3.8
    srt - To parse and manipulate the subtitles.
    sqlite3 - To store the subtitles and their embeddings in a database.
    openai - To generate embeddings for each subtitle, you will need an API key
    
Installing

    Clone the repository to your local machine.

$bash git clone https://github.com/thegoodwei/footagesorting.git

    Install the required packages.

pip install -r openai srt pysqlite3 numpy wheel pyjson

Functionality

    combine_subs - Combines multiple subtitles into a single block if they fall within a certain duration limit.
    write_to_file - Writes the combined subtitles to a file in the .srt format.
    db_setup - Sets up a database to store the subtitles and their embeddings.
    load_subtitles - Loads the subtitles into the database.
    get_embeddings - Generates embeddings for each subtitle and stores them in the database.

Usage

 Run the program and you will be prompted for the following:
    Enter the title of the project.
    Enter the name of the transcript file in the .srt format.
    Enter the duration for each block of text (in seconds).
    Enter the number of soundbites to be shown.
    Enter the keywords to prompt for relevancy ranking.

The script will then run and store the relevant subtitles along with their embeddings in the database.

python main.py --input_file [INPUT_SRT_FILE] --prompt [PROMPT] --length [LENGTH] --output_file [OUTPUT_SRT_FILE]
Contributing

Feel free to submit pull requests or issues. Any contribution is appreciated.
Author

[theGoodWei]

Open Source

import openai
openai.api_key = "YOUR-OPENAI-API-KEY"
import srt
import sqlite3
import json
import random
import time
import numpy as np
import re
import os
import time
import requests 
 ts = time.time()

if os.path.exists("temp.srt"):
        os.remove("temp.srt")
if os.path.exists("TempDatabase_Subtitling.db"):
        os.remove("TempDatabase_Subtitling.db")

    # Settings
projecttitle = input("title project :: ")
srt_input_name = input("transcript.srt :: ")
input_subtitle_file = srt_input_name # "transcript.srt"
output_srt_file = projecttitle + "_relevant_subs.srt"
db_file = "TempDatabase_Subtitling.db"
edlfile = projecttitle + "relevantfootage_timeline.edl"
delay = 60.00 / 300         #60.00 / rate_limit_per_minute

quotelength = int(input("apx duration for each block of text, in seconds?"))
this_many_quotes = int(input("How many soundbites to show?       "))
user_prompt = input("Keywords to prompt for relevancy ranking:     ")
print("initiating..............................................................")
subtitle_file = "temp.srt"
def combine_subs(input_subtitle_file):
    with open(input_subtitle_file, 'r') as f:
        srt_text = f.read()
    f.close()
    subs = list(srt.parse(srt_text))
    combined_subs = []
    i = 0
    while i < len(subs):
        sub = subs[i]
        start = sub.start.total_seconds()
        end = sub.end.total_seconds()
        text = sub.content
        count = 0
        j = i + 1
        while (j < len(subs)) and ((end - start) < (quotelength*2)):
            addsub = subs[j]
            text += ' ' + addsub.content
            end = addsub.end.total_seconds()
            j += 1
        combined_subs.append((srt.Subtitle(index=(count), start=srt.timedelta(seconds=start), end=srt.timedelta(seconds=end), content=text)))
        count +=1
        i=j+1
    return write_to_file((combined_subs), subtitle_file)

def write_to_file(combined_subs, file_path):
#clear files to write project outputs
    if os.path.exists("temp.srt"):
        os.remove("temp.srt")
    with open(file_path, 'w') as f:
        f.write(srt.make_legal_content(srt.compose(combined_subs)))
    subtitle_file = file_path
    f.close()
    print("written to file")
    return file_path

def db_setup(db_file):
    # Connect to the database
    conn = sqlite3.connect(db_file)
    c = conn.cursor()
    # Create the table if it doesn't exist
    c.execute("CREATE TABLE IF NOT EXISTS subtitles (start REAL, end REAL, text TEXT, srt_file TEXT, summary TEXT, embeddings TEXT)")
    # Commit the changes
    conn.commit()
    # Close the connection
    conn.close()

def load_subtitles(subtitle_file, db_file):
    conn = sqlite3.connect(db_file)
    c = conn.cursor()
    # Open the subtitle file
    f = open(subtitle_file,"r")
    srt_text = f.read()
    f.close()
    subtitles = srt.parse(srt_text)
    # Parse the subtitles into list format
    # combine subs does not work???
    for sub in subtitles:
        # Get the start and end times, and convert them to seconds from the start of the file
        start = sub.start.total_seconds()
        end = sub.end.total_seconds()
        # Get the text of the subtitle
        text = sub.content
        # Insert into the database
        c.execute("INSERT INTO subtitles VALUES (?,?,?,?,?,?)", (start, end, text, subtitle_file, "None", "None"))
        print(text)
    # Commit the changes
    conn.commit()
    # Close the connection
    conn.close()

def get_embeddings(subtitle_file, db_file):
    # Connect to the database
    conn = sqlite3.connect(db_file)
    c = conn.cursor()
    # Get the subtitles from the database
    c.execute("SELECT start,end,text FROM subtitles WHERE srt_file = ? and embeddings='None'", (subtitle_file,))
    subtitles = c.fetchall()
    # Get the text of the subtitles
    for time_start,time_end,sub_text in subtitles:
        delayed_completion(delay_in_seconds=delay)
        response = openai.Embedding.create(model="text-embedding-ada-002", input=sub_text)
        embedding = response["data"][0]["embedding"]
        c.execute("UPDATE subtitles SET embeddings = ? WHERE start = ? AND end = ? AND srt_file = ?", (json.dumps(embedding), time_start, time_end, subtitle_file))
        # Commit the changes
        conn.commit()
#        print(format_time(time_end))
    # Close the connection
    conn.close()
    print(" ")
    print("Embedded meanings and associations found for every clip.")
    print(" ...")

def search_database(subtitle_file, db_file, query, top_n=int(this_many_quotes)):
    print("Search has begun:") 
    # Connect to the database
    conn = sqlite3.connect(db_file)
    c = conn.cursor()
    # Create a temporary database in memory to store the results
    memconn = sqlite3.connect(":memory:")
    memc = memconn.cursor()
    memc.execute("CREATE TABLE IF NOT EXISTS subtitles (start REAL, end REAL, text TEXT, similarity_score REAL)")
    memconn.commit()
    # Delay for rate limiter
    delayed_completion(delay_in_seconds=delay)
    # Get the embeddings for the query
    response = openai.Embedding.create(model="text-embedding-ada-002", input=query)
    query_embedding = response["data"][0]["embedding"]
    # Get the subtitles from the database
    c.execute("SELECT start,end,text,embeddings FROM subtitles WHERE srt_file = ? and embeddings != 'None'", (subtitle_file,))
    subtitles = c.fetchall()
    # Close the connection
    conn.close()
    # Get the text of the subtitles 
    similarity_avg = 0
    avgsimilarity = [1.00, 1]  #[total, count]
    with open(("log.txt"), 'w') as f:
        f.write(" ")
        f.close()
    for time_start,time_end,sub_text,sub_embedding in subtitles:
        delayed_completion(delay_in_seconds=delay)
        # Get the embedding for the subtitle
        sub_embedding = json.loads(sub_embedding)
        # Calculate the cosine similarity
        similarity = np.dot(query_embedding, sub_embedding) / (np.linalg.norm(query_embedding) * np.linalg.norm(sub_embedding))
        # Print above avg results
        avgsimilarity[0] += similarity #total
        avgsimilarity[1] +=1 #count
        similarity_avg = avgsimilarity[0]/avgsimilarity[1]  #total/count
        if (similarity > (similarity_avg)):
            with open("log.txt", 'r') as f:
                log_text = f.read()
            f.close()
            with open(("log.txt"), 'w') as f:
                f.write(log_text)
                f.write("\n")
                f.write(srt.timedelta_to_srt_timestamp(srt.timedelta(seconds=time_start)))
                f.write( " --> ")
                f.write(srt.timedelta_to_srt_timestamp((srt.timedelta(seconds=time_end))))
                f.write("\n")
                f.write(sub_text)
                f.write(str(similarity))
                f.write(" \n ")
            f.close()
            print(srt.timedelta_to_srt_timestamp(srt.timedelta(seconds=time_start)))
            print("  -  ")
            print(srt.timedelta_to_srt_timestamp(srt.timedelta(seconds=time_end)))
            print(sub_text)
            print(similarity)
        # Insert data into the temporary database
        memc.execute("INSERT INTO subtitles VALUES (?,?,?,?)", (time_start, time_end, sub_text, similarity))
        memconn.commit()
    f.close()
    print(".....................................................")
    print(".....................................................")
    print(".....................................................")
    # Get the top n results
    memc.execute("SELECT start,end,text,similarity_score FROM subtitles ORDER BY similarity_score DESC LIMIT ?", (top_n,))
    results = memc.fetchall()
    # Print the results
    selected_subs = []
    index = 1
    for time_start,time_end,sub_text,similarity_score in results:
        # Convert time_start and time_end back to timedelta objects
        time_start = srt.timedelta(seconds=time_start)
        time_end = srt.timedelta(seconds=time_end)
        #sel content = sub_text # "{similarity_score} \n {sub_text}"
        #selectofsub = find_most_complete_section(sub_text, user_prompt)
        # Print the results
        print(time_start, time_end, "\n", sub_text, "\n", "=", similarity_score)
        selected_subs.append(srt.Subtitle(index=index, start=time_start, end=time_end, content=sub_text))
        index+=1
    #Print selects to file, with details to terminal
    write_to_file(selected_subs, output_srt_file)
    print(".....................................................")
    print(".....................................................")
    print("prompt : transcript")
    print("similarity score:")
    print(similarity_avg)
    print("-")
    print("prompted by the keywords:")
    print(user_prompt)
    print(".....................................................")
    print("Found")
    print(this_many_quotes)
    print("quotes")
    print("----")
    print("runtime:")
    print(str(format_time(int(quotelength)*int(this_many_quotes)))) 
    print("----")
    print("...........................")
    print("EDL file ready for NLE import:")
    print(output_srt_file)
    print("-")
    print("more above-average clips found at at ./log.txt")

def generate_edl(output_srt_file):
    fullvideo = "fullvideo.mp4"
    with open(output_srt_file, 'r') as f:
        srt_text = f.read()
        subs = list(srt.parse(srt_text))
        subtimes = [] # make this a tuple of start_time and end_time

        for sub in subs:
            start = sub.start.total_seconds()
            end = sub.end.total_seconds()
            #subtimes.append(start_time=(timedelta(seconds=start)), end_time=(timedelta(seconds=end)))
            start_timecode = srt.timedelta(seconds=start)
            end_timecode = srt.timedelta(seconds=end)
            subtimes.append((start_timecode, end_timecode))
        f.close()
    id0 =0
    id00=0
    id000=0
    i = 0
    with open(edlfile, "w") as f:
        f.write("TITLE: fullvideo \n FCM: NON-DROP FRAME \n \n")
        #f.write(f"* FROM CLIP NAME: {fullvideo}\n")
        cursor = ""#?
        for i, (start_timecode, end_timecode) in enumerate(subtimes):
            cut_in = start_timecode
            cut_out = end_timecode
            print(cut_in)

            id0+=1
            if id0 >9:
                id0 = 0
                id00 += 1
            if id00>9:
                id00=0
                id000+=1
            print("-->")
            f.write(str(id000))
            f.write(str(id00))
            f.write(str(id0))
            f.write("  AX       AA/V  C        ")
            if id0 == 1:
                print("\n")
                print(str(convert_time(cut_out)))
                cursor = cut_out - cut_in
                f.write(str(convert_time(cut_in)))
                f.write(" ")
                f.write(str(convert_time(cut_out)))
                f.write(" ")
                f.write("00:00:00:00")
                f.write(" ")
                f.write(str(convert_time(cursor)))
                f.write("\n")
            else:
                f.write(str(convert_time(cut_in)))
                f.write(" ")
                f.write(str(convert_time(cut_out))) 
                f.write(" ")
                f.write(str(convert_time(cursor)))
                f.write(" ")
                f.write(str(convert_time(cursor+(cut_out - cut_in))))
                cursor += (cut_out -cut_in)

                f.write("\n")
            f.write("* FROM CLIP NAME: fullvideo.mp4")
            f.write("\n")
            f.write("\n")
            print(str(convert_time(cut_out)))
            print("\n\n")
        print(edlfile)
    f.close()

def find_most_complete_section(text, keywords):
    print(".")
    time.sleep(5)
    print("..")
    time.sleep(5)
    print("...")
    completions = openai.Completion.create(
        engine="text-davinci-002",
        prompt=f"find the most syntatically complete representation of the main idea enclosed in the following text, and return the main section verbatem: {text}",
        max_tokens=1024,
        n=1,
        stop=None,
        temperature=0.2,
    )
    message = completions.choices[0].text
    return message

def convert_time(time):
    hours, remainder = divmod(time.seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    frames = int(time.microseconds / 1000000 * 30)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}:{frames:02d}"
# Define a function that adds a delay to a Completion API call
def delayed_completion(delay_in_seconds: float = 1):
    """Delay a completion by a specified amount of time."""
    print(".....")
    time.sleep(delay_in_seconds)

def format_time(seconds):
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    seconds = seconds % 60
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{int(seconds * 1000) % 1000:03d}"

if __name__ == "__main__":
    db_setup(db_file)
    combine_subs(input_subtitle_file)
    load_subtitles(subtitle_file, db_file)
    get_embeddings(subtitle_file, db_file)
    search_database(subtitle_file, db_file, user_prompt)
    generate_edl(output_srt_file)


#Works in Progress

#def determine_completeness(text):
#    response = openai.Completion.create(
#        engine="davinci",
#        prompt=f"{text}",
#        max_tokens=1024,
#        n=1,
#        stop=None,
#        temperature=0.5,
#    )


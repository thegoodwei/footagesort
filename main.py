import openai
openai.api_key = "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"
import srt
import sqlite3
import json
import random
import time
import numpy as np
import sys, os, re, edl, tempfile, argparse
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "MAX_SPLIT_ALLOC_SIZE=void"
import torch
import time
import requests 
import timecode
from pathlib import Path 
import whisperx
from pydub import AudioSegment
import logging
import subprocess
from moviepy.editor import *
from pymediainfo import MediaInfo

project_title = "title"

#import whisper
#import whisper.utils
#from whisper.utils import write_srt 
project_title = input("title project :: ")

quotelength = int(input("apx duration for each block of text, in seconds?"))
this_many_quotes = int(input("How many soundbites to show?       "))
user_prompt = input("Keywords to prompt for relevancy ranking:     ")
user_provided_file =  input("input filename .mp4 .m4a .mp3 .mov : ")
if "srt" in user_provided_file:
    input_subtitle_file = user_provided_file
    #skip the transcription

ts = time.time()
if os.path.exists("temp.srt"):
        os.remove("temp.srt")
if os.path.exists("TempDatabase_Subtitling.db"):
        os.remove("TempDatabase_Subtitling.db")

    # Settings
input_subtitle_file = "transcript.srt"
subtitle_file = "temp.srt"
output_srt_file = project_title + ".srt"
db_file = "TempDatabase_Subtitling.db"
edl_file = "relevantfootage_timeline.edl"
delay = 60.00 / 500         #60.00 / rate_limit_per_minute
#def user_setup():
#input_subtitle_file = input("transcript.srt :: ")
#with open(input_subtitle_file, 'r') as f:
#    f.read()
#with open(subtitle_file, 'r') as f:
#    f.read()
output_srt_file = project_title + "_relevant_subs.srt"
db_file = project_title + "TempDatabase_Subtitling.db"
edl_file = project_title + "relevantfootage_timeline.edl"
editedoutput = project_title + "_"  + user_provided_file
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

def write_to_file(newsubs, file_path):
    #clear files to write project outputs
    if os.path.exists("temp.srt"):
        os.remove("temp.srt")
    with open(file_path, 'w') as f:
        f.write(srt.make_legal_content(srt.compose(newsubs)))
    subtitle_file = file_path
    f.close()
    print("written to file")
    return file_path

def create_srt_transcript(input_file: str, output_file: str, device: str = "cuda") -> None:
    """
    Create an srt transcript from an audio file.
    Args:
    - input_file (str): the path to the input audio file
    - output_file (str): the path to the output srt file
    - device (str): the device to use for processing, either "cuda" or "cpu" (default "cuda")
    Returns:
    None
    """
    input_audio = "wavfile.wav"
    with open(input_audio, 'w+') as f:
        f.close()
    print("creating audio file...")
    # input_audio = input_file[:-4] + ".wav"
    # Convert the input file
    audio = AudioSegment.from_file(input_file, format=input_file.split(".")[-1])
    audio.export(input_audio, format="wav")

    print("conversion to wav is successful!")
    # Export the audio to the .wav format

    # Load the original whisper model
    print("Standby while we load whisper")
    try:
        model = whisperx.load_model("medium", device)
        result = model.transcribe(input_audio) #was input_audio if converted filetype
        # Load the alignment model and metadata
        model_a, metadata = whisperx.load_align_model(language_code=result["language"], device=device)
        print("aligning timecode")
        # Align the whisper output
        result_aligned = whisperx.align(result["segments"], model_a, metadata, input_audio, device)
    except Exception as e:
        logging.error("Failed to align whisper output: %s", e)
        return
    #print(result["segments"]) # before alignment
    #print("_______word_segments")
    #print(result_aligned["word_segments"]) # after alignment
    #write_srt(result_aligned[segment], TextIO.txt)

#    audio_basename = Path(audio_path).stem 
#    with open(Path(output_dir) / (audio_basename + ".srt"), "w", encoding="utf-8") as srt: 
#        write_srt(result_aligned["segments"], file=srt)

    # Create the .srt transcript
    srt_transcript = []
    i=1
    for (segment) in result_aligned["word_segments"]:
#       start = segment.start.total_seconds()
#       end = segment.end.total_seconds()
        start_time = srt.timedelta(seconds=int(segment['start']))
        end_time = srt.timedelta(seconds=int(segment['end']))
        start_time = srt.timedelta(seconds=int(start['start'].total_seconds()))
#       end_time = srt.timedelta(seconds=int(end['end'].total_seconds()))
        #start_time = srt.timestamp(segment['start'])
        #end_time = srt.timestamp(segment['end'])

        text = segment['text'] #.strip().replace('-->', '->')
        srt_transcript.append(srt.Subtitle(index=i, start=start_time, end=end_time, content=text))
        i+=1
    # Write the .srt transcript to a file
  #  with open(output_file, "w", encoding="utf-8") as f:
   #     srt.write_srt(srt_transcript, f)
    return write_to_file((srt_transcript), input_subtitle_file)

#def write_srt(transcript: Iterator[dict], file: TextIO): 
    #     """ 
    #     Write a transcript to a file in SRT format. 
    #  
    #     Example usage: 
    #         from pathlib import Path 
    #         from whisper.utils import write_srt 
    #  
    #         result = transcribe(model, audio_path, temperature=temperature, **args) 
    #  
    #         # save SRT 
    #         audio_basename = Path(audio_path).stem 
    #         with open(Path(output_dir) / (audio_basename + ".srt"), "w", encoding="utf-8") as srt: 
    #             write_srt(result["segments"], file=srt) 
    #     """ 
    #     for i, segment in enumerate(transcript, start=1): 
    #         # write srt lines 
    #         print( 
    #             f"{i}\n" 
    #             f"{format_timestamp(segment['start'], always_include_hours=True, decimal_marker=',')} --> " 
    #             f"{format_timestamp(segment['end'], always_include_hours=True, decimal_marker=',')}\n" 
    #             f"{segment['text'].strip().replace('-->', '->')}\n", 
    #             file=file, 
    #             flush=True, )
    # 

    #if bool(input("'True' to transcribe from an audio file, 'False' to import from an .srt file : ")):
#input_subtitle_file = input("Subtitlefile.srt")

def find_complete_section(text) -> bool:
    brake = False
    # if has_punctuation_in_last_three_characters(text): return True
    last_three_characters = text[-4:]
    for char in last_three_characters:
        if char in ".,?!:":
            print(".,!?")
            brake = True
            break
    if (brake == False):
        completions = openai.Completion.create(
        engine="text-babbage-001",
        prompt=f"[Determine 'True' or 'False'] does the last phrase in this text end at a stopping point? {text}",
        max_tokens=256,
        n=1,
        stop=None,
        temperature=0.25,
        )
        message = completions.choices[0].text
        print("Babbage:")
        print(message)
        if "True" in message:
            return True
        else:
            if "False" in message:
                return False
            else:
                if "true" in message:
                    return True
                else:
                    if "false" in message:
                        return False
                    else:
                        if "yes" in message:
                            return True
                        else:
                            if "Yes" in message:
                                return True
                            else:
                                if "No" in message:
                                    return False
                                else:
                                    if "no" in message:
                                        return False
                                    else:
                                        if "does not" in message:
                                            return False
                                        else:
                                            if "finishes on a complete" in message:
                                                return True
                                            else:
                                                if "ends on a complete" in message:
                                                    return True
                                                else:
                                                    if "does" in message:
                                                        return True
                                                    else:
                                                        if "finishes with a complete" in message:
                                                            return True
                                                        else:
                                                            return False
    else:
        return brake

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
        while (j < len(subs)) and ((end - start) < (quotelength)):
            addsub = subs[j]
            text += ' ' + addsub.content
            end = addsub.end.total_seconds() 
            j += 1
        iscomplete = find_complete_section(text)
        print(str(iscomplete))
        if (iscomplete == True) or (iscomplete == "True"):# or ('true' in iscomplete)):
            combined_subs.append((srt.Subtitle(index=(count), start=srt.timedelta(seconds=start), end=srt.timedelta(seconds=end), content=text)))
            count +=1
            i=j+1
            print("Sentence iscomplete first time!")
        else:
            print("Sentance not complete, starting loop:")
            while (j<len(subs)):
                print (i)
                print(j)
                if (j==len(subs)):
                    break
                addsub = subs[j]
                text += ' ' + addsub.content
                end = addsub.end.total_seconds() 
                iscomplete = find_complete_section(text)
                print(str(iscomplete))
                if (iscomplete == True) or (iscomplete == "True"): # ('true' in iscomplete)):
                    break
                j += 1
                print("loop")
            combined_subs.append((srt.Subtitle(index=(count), start=srt.timedelta(seconds=start), end=srt.timedelta(seconds=end), content=text)))
            count +=1
            i=j+1
    return write_to_file((combined_subs), subtitle_file)

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
        # print(format_time(time_end))
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
    avgsimilarity = [0.00, 0]  #[total, count] #could be 1/1 for above avg if long script
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
    memc.execute("SELECT start,end,text,similarity_score FROM subtitles ORDER BY similarity_score DESC LIMIT ?", (top_n,)) #was ORDER BY similarity_score
    results = memc.fetchall()
    # Print the results
    selected_subs = []
    index = 1
    for time_start,time_end,sub_text,similarity_score in results:
        # Convert time_start and time_end back to timedelta objects
        time_start = srt.timedelta(seconds=time_start)
        time_end = srt.timedelta(seconds=time_end)
        #sel content = sub_text # "{similarity_score} \n {sub_text}"
        #selectofsub = find_complete_section(sub_text, user_prompt)
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
    print("Searched for")
    print(this_many_quotes)
    print("quotes")
    print("found")
    print(len(results))
    print("----")
    print("intended runtime:")
    print(str(format_time(int(quotelength)*int(this_many_quotes)))) 
    print("----")
    print("...........................")
    print("EDL file ready for NLE import:")
    print(output_srt_file)
    print("-")
    print("more above-average clips found at at ./log.txt")

def generate_edl(output_srt_file):
    fullvideo = user_provided_file #"fullvideo.mp4"
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
    with open(edl_file, "w") as f:
        f.write("TITLE: " + project_title + " \n" + "FCM: NON-DROP FRAME \n \n")
        f.write(f"* FROM CLIP NAME: {fullvideo}\n")
        cursor = 0 #?
       # seconds, not timecode
        for i, (start_timecode, end_timecode) in enumerate(subtimes):
            cut_in = start_timecode
            cut_out = end_timecode
            print(cut_in)
            #Converted into timecode with frames
            #print(str(convert_time(cut_in)))

            id0+=1
            if id0 >9:
                id0 = 0
                id00 += 1
            if id00>9:
                id00=0
                id000+=1

           # print("-->")
            f.write(str(id000))
            f.write(str(id00))
            f.write(str(id0))
            f.write("  AX       AA/V  C        ")
            if id0 == 1:
              #  print("\n")
               # print(str(convert_time(cut_out)))
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
                cursor += (cut_out - cut_in)

                f.write("\n")
           # f.write("* FROM CLIP NAME: fullvideo.mp4")
            f.write("\n")
            # f.write("\n")
           # print(str(convert_time(cut_out)))
           # print("\n\n")
        print(str(convert_time(cursor+(cut_out - cut_in))))
        print(edl_file)
    f.close()

def convert_time(time):
    hours, remainder = divmod(time.seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    frames = int(time.microseconds / 1000000 * 30)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}:{frames:02d}"
def format_time(seconds):
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    seconds = seconds % 60
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{int(seconds * 1000) % 1000:03d}"
# adds a delay to a Completion API call if rate limited
def delayed_completion(delay_in_seconds: float = 1):
    """Delay a completion by a specified amount of time."""
    #print(".....")
    time.sleep(delay_in_seconds)

#def apply_edits(input_mp3, edl_file, editedoutput):
#    with open(edl_file, "r") as f:
#        lines = f.readlines()
#
#    # Convert the .edl file into a list of FFmpeg commands
#    ffmpeg_commands = []
#    for line in lines:
#        if line.startswith("TITLE:"):
#            continue
#        if line.startswith("FCM:")
#            continue
#        if line.startswith("* FROM CLIP NAME"):
#            continue
#        if not line.strip():
#            continue
#        start_in, end_in, start_out, end_out = line.strip().split()[3:]
#        #start_time, end_time = line.strip().split("\t")
#        ffmpeg_command = [
#            "ffmpeg",
#            "-i", input_mp3,
#            "-ss", start_time,
#            "-to", end_time,
#            "-c", "copy",
#            "part.mp3"
#        ]
#        ffmpeg_commands.append(ffmpeg_command)
#    # Concatenate the parts
#    with open("filelist.txt", "w") as f:
#        for i, cmd in enumerate(ffmpeg_commands):
#            f.write("file 'part_{}.mp3'\n".format(i))
#
#    concat_command = [
#        "ffmpeg",
#        "-f", "concat",
#        "-safe", "0",
#        "-i", "filelist.txt",
#        "-c", "copy",
#        editedoutput
#    ]


class Edit(object):
    def __init__(self, time1, time2, action):
        self.time1 = str(time1)
        self.time2 = str(time2)
        self.action = str(action)
class EDL(object):
    def __init__(self, edlfile):
        self.edits = []
        self.edlfile = edlfile

        if os.path.exists(self.edlfile) == False:
            open(self.edlfile, 'a').close()
        else:
            with open(self.edlfile) as f:
                for line in f.readlines():
                    if len(line.split()) == 3:
                        self.edits.append(Edit(line.split()[0], line.split()[1], line.split()[2].split('\n')[0]))
                    elif len(line.split()) == 2:
                        self.edits.append(Edit(line.split()[0], line.split()[1], "-"))
    def sort(self):
        self.edits.sort(key=lambda x: float(x.time1))
                
    def save(self):
        self.sort()
        with open(self.edlfile, 'w') as f:
            for edit in self.edits:
                f.writelines(str(edit.time1)+"      "+str(edit.time2)+"      "+edit.action+"\n")
                
    def add(self, time1, time2, action):
        self.edits.append(Edit(time1, time2, action))
        self.sort()

def render(user_provided_file, estruct, editedoutput, videoBitrate="2000k", audioBitrate="400k", threadNum=2, ffmpegPreset="medium", vcodec=None, acodec=None, ffmpeg_params=None, writeLogfile=False):
    clipNum = 1
    global prevTime
    prevTime = 0
    actionTime = False
    v = VideoFileClip(user_provided_file)
    duration = v.duration
    clips = v.subclip(0,0) #blank 0-time clip
    for edit in estruct.edits:
        if (edit.startswith("TITLE:") or edit.startswith("FCM:") or edit.startswith("* FROM CLIP NAME")): #or not edit.strip():
            continue
        else:
            nextTime = float(edit.time1)
            time2 = float(edit.time2)
            action = edit.action

            if nextTime > duration:
                nextTime = duration

            if prevTime > duration:
                prevTime = duration

            clip = v.subclip(prevTime,nextTime)
            clips = concatenate([clips,clip])
            print("created subclip from " + str(prevTime) + " to " + str(nextTime))
            prevTime = nextTime
            nextTime = time2
            if action == "1":
                # Muting audio only. Create a segment with no audio and add it to the rest.
                clip = VideoFileClip(user_provided_file, audio = False).subclip(prevTime,nextTime)
                clips = concatenate([clips,clip])
                print("created muted subclip from " + str(prevTime) + " to " + str(nextTime))
                # Advance to next segment time.
                prevTime = nextTime
            elif action == "0":
                #Do nothing (video and audio cut)
                print("Cut video from "+str(prevTime)+" to "+str(nextTime)+".")
                prevTime = nextTime
            elif action == "2":
                # Cut audio and speed up video to cover it.
                #v = VideoFileClip(videofile)
                # Create two clips. One for the cut segment, one immediately after of equal length for use in the speedup.
                s1 = v.subclip(prevTime,nextTime).without_audio()
                s2 = v.subclip(nextTime,(nextTime + s1.duration))
                # Put the clips together, speed them up, and use the audio from the second segment.
                clip = concatenate([s1,s2.without_audio()]).speedx(final_duration=s1.duration).set_audio(s2.audio)
                clips = concatenate([clips,clip])
                print("Cutting audio from "+str(prevTime)+" to "+str(nextTime)+" and squeezing video from "+str(prevTime)+" to "+str(nextTime + s1.duration)+" into that slot.")
                # Advance to next segment time (+time from speedup)
                prevTime = nextTime + s1.duration
            else:
                # No edit action. Just put the clips together and continue.
                clip = v.subclip(prevTime,nextTime)
                clips = concatenate([clips,clip])
                # Advance to next segment time.
                prevTime = nextTime


    videoLength = duration
    if prevTime > duration:
        prevTime = duration

    if ffmpeg_params != None:
        fparams = []
        for x in ffmpeg_params.split(' '):
            fparams.extend(x.split('='))
    else:
        fparams = None

    clip = v.subclip(prevTime,videoLength)
    print("created ending clip from " + str(prevTime) + " to " + str(videoLength))
    clips = concatenate([clips,clip])
    clips.write_videofile(editedoutput, codec=vcodec, fps=30, bitrate=videoBitrate, audio_bitrate=audioBitrate, audio_codec=acodec, ffmpeg_params=fparams, threads=threadNum, preset=ffmpegPreset, write_logfile=writeLogfile)


def parse_edl_edits(user_provided_file=user_provided_file, edl_file=edl_file, editedoutput=editedoutput): #user_provided_file, edl_file, editedoutput
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--threads", type=int, help="Number of CPU threads to use.")
    parser.add_argument("-p", "--preset", choices=["ultrafast", "superfast", "fast", "medium", "slow", "superslow"], help="FFMPEG preset to use for optimizing the compression. Defaults to 'medium'.")
    parser.add_argument("-vb", "--videobitrate", help="Video bitrate setting. Auto-detected from original video unless specified.")
    parser.add_argument("-ab", "--audiobitrate", help="Audio bitrate setting. Auto-detected from original video unless specified.")
    parser.add_argument("-vc", "--vcodec", help="Video codec to use.")
    parser.add_argument("-ac", "--acodec", help="Audio codec to use.")
    parser.add_argument("-fp", "--ffmpegparams", help="Additional FFMpeg parameters to use. Example: '-crf=24 -s=640x480'.")
    args = parser.parse_args()
#    with open(edl_file, "r") as f:
#        lines = f.readlines()
    # Convert the .edl file into a list of FFmpeg commands    ffmpeg_commands = []
#    for line in lines:
#        if line.startswith("TITLE:"):
#            continue
#        if line.startswith("FCM:"):
#            continue
#        if line.startswith("* FROM CLIP NAME"):
#            continue
#        if not line.strip():
#            continue
#    estruct = EDL(lines)
    estruct = EDL(edl_file)
    videoBitrate = ""
    audioBitrate = ""

    if args.threads == None:
        threadNum = 2
    else:
        threadNum = args.threads

    if args.preset == None:
        ffmpegPreset = "medium"
    else:
        ffmpegPreset = args.preset

 #   mi = MediaInfo.parse(user_provided_file)
    if args.videobitrate == None:
 #       videoBitrate = str(int(mi.tracks[1].bit_rate / 1000)) + "k"
 #       print("Using original video bitrate: "+videoBitrate)
 #   else:
        videoBitrate = "4500k" #args.videobitrate
 #       if videoBitrate[-1] != 'k':
 #           videoBitrate = videoBitrate+'k'
    if args.audiobitrate == None:
 #       try:
 #           audioBitrate = str(int(mi.tracks[2].bit_rate / 1000)) + "k"
 #       except TypeError:
 #           audioBitrate = str(int(int(mi.tracks[2].bit_rate.split(' / ')[1]) / 1000)) + "k"
#
 #       print("Using original audio bitrate: "+audioBitrate)
 #   else:
        audioBitrate = "320k" #args.audiobitrate
  #      if audioBitrate[-1] != 'k':
  #          audioBitrate = audioBitrate+'k'

    render(user_provided_file, estruct, editedoutput, videoBitrate, audioBitrate, threadNum=threadNum, vcodec=args.vcodec, acodec=args.acodec, ffmpeg_params=args.ffmpegparams, ffmpegPreset=ffmpegPreset)


if __name__ == "__main__":
    input_subtitle_file = create_srt_transcript(user_provided_file, "_xtranscript.srt" )
    editedoutput += user_provided_file
    with open(editedoutput, "w+") as f:
        empty = f.read()
    db_setup(db_file)
    combine_subs(input_subtitle_file)
    load_subtitles(subtitle_file, db_file)
    get_embeddings(subtitle_file, db_file)
    search_database(subtitle_file, db_file, user_prompt)
    generate_edl(output_srt_file)
    parse_edl_edits(user_provided_file, edl_file, editedoutput)
    print(editedoutput)
    #apply_edits(user_provided_file, edl_file, editedoutput)
    #command = ['ffmpeg', '-i', user_provided_file, '-f', 'edl', '-i', edl_file, '-codec', 'copy', editedoutput]
    #subprocess.call(command)

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

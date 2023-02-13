class EDL:
    def __init__(self, project_name = "", video_files = [], audio_files = [], timecode = "00:00:00:00", edits = [], play_times = [], clip_names = [], fps = 29.97, codec = "uncompressed", resolution = "HD", aspect_ratio = "16:9", total_clips = 0, total_runtime = 0, subtitles_srt = "", metadata = ""):
        self.project_name = project_name
        self.video_files = video_files
        self.audio_files = audio_files
        self.timecode = timecode
        self.edits = edits
        self.play_times = play_times
        self.clip_names = clip_names
        self.fps = fps
        self.codec = codec
        self.resolution = resolution
        self.aspect_ratio = aspect_ratio
        self.total_clips = total_clips
        self.total_runtime = total_runtime
        self.subtitles_srt = subtitles_srt
        self.metadata = metadata

        self.edlfile = edlfile

        if (os.path.exists(self.edlfile) == False):
            open(self.edlfile, 'a').close()
        else:
            with open(self.edlfile) as f:
                for line in f.readlines():
                    if len(line.split()) == 3:




    def parse_timecode(self, timecode_str):
        timecode_parts = timecode_str.split(":")
        frames = int(timecode_parts[3]) * self.fps
        seconds = int(timecode_parts[2])
        minutes = int(timecode_parts[1])
        hours = int(timecode_parts[0])
        total_frames = int(frames + (seconds * self.fps) + (minutes * self.fps * 60) + (hours * self.fps * 60 * 60))
        return total_frames

    def add_edit_from_srt(self, start_time, end_time, filename):
        start_frames = self.parse_timecode(start_time)
        end_frames = self.parse_timecode(end_time)
        self.add_edit(start_frames, end_frames, filename)

    def add_edit(self, start_frames, end_frames, filename):
        self.edits.append((start_frames, end_frames, filename))
        self.total_clips += 1

    def to_edl_string(self):
        edl_string = ""
        edl_string += "TITLE: " + self.project_name + "\n"
        edl_string += "FCM: NON-DROP FRAME\n"
        for edit in self.edits:
            start_frames = edit[0]
            end_frames = edit[1]
            filename = edit[2]
            start_timecode = self.frames_to_timecode(start_frames)
            end_timecode = self.frames_to_timecode(end_frames)
            edl_string += "00:00:00:00 " + start_timecode + " " + filename + " " + start_timecode + " " + end_timecode + " " + filename + "\n"
        return edl_string

     def frames_to_timecode(self, frames):
        total_seconds = frames / self.fps
        hours = int(total_seconds / 3600)
        minutes = int((total_seconds - hours * 3600) / 60)
        seconds = int(total_seconds - hours * 3600 - minutes * 60)
        milliseconds = int((total_seconds - int(total_seconds)) * 1000)
        return f"{hours:02}:{minutes:02}:{seconds:02},{milliseconds:03}"

def parse_srt(srt_file):
    if not isinstance(srt_file, str):
        raise TypeError("Invalid input type for srt_file, expected string, got {}".format(type(srt_file)))
    srt_lines = srt_file.split('\n')
    if len(srt_lines) == 0:
        raise ValueError("Invalid input for srt_file, no lines found")
    timecodes = []
    current_line = 1
    while current_line < len(srt_lines):
        if srt_lines[current_line].strip() == '':
            current_line += 1
            continue
        try:
            start, end = srt_lines[current_line].split(' --> ')
        except ValueError as e:
            raise ValueError("Invalid format for timecode in line {} of srt_file, expected 'start --> end' format".format(current_line)) from e
        start = start.strip()
        end = end.strip()
        timecodes.append((start, end))
        current_line += 3
    return timecodes

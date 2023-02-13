
from sqlalchemy import create_engine, Column, Integer, String, Float, ForeignKey
from sqlalchemy.orm import relationship, backref
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()

class EDL(Base):
    __tablename__ = 'edl'
    id = Column(Integer, primary_key=True)
    project_name = Column(String)
    timecode = Column(String)
    fps = Column(Float)
    codec = Column(String)
    resolution = Column(String)
    aspect_ratio = Column(String)
    total_clips = Column(Integer)
    total_runtime = Column(Float)
    subtitles_srt = Column(String)
    metadata = Column(String)
    
    video_files = relationship('VideoFile', backref='edl', cascade="all, delete-orphan")
    audio_files = relationship('AudioFile', backref='edl', cascade="all, delete-orphan")
    edits = relationship('Edit', backref='edl', cascade="all, delete-orphan")
    play_times = relationship('PlayTime', backref='edl', cascade="all, delete-orphan")
    clip_names = relationship('ClipName', backref='edl', cascade="all, delete-orphan")

class VideoFile(Base):
    __tablename__ = 'video_file'
    id = Column(Integer, primary_key=True)
    file_path = Column(String)
    edl_id = Column(Integer, ForeignKey('edl.id'))

class AudioFile(Base):
    __tablename__ = 'audio_file'
    id = Column(Integer, primary_key=True)
    file_path = Column(String)
    edl_id = Column(Integer, ForeignKey('edl.id'))

class Edit(Base):
    __tablename__ = 'edit'
    id = Column(Integer, primary_key=True)
    time1 = Column(Float)
    time2 = Column(Float)
    action = Column(String)
    edl_id = Column(Integer, ForeignKey('edl.id'))

class PlayTime(Base):
    __tablename__ = 'play_time'
    id = Column(Integer, primary_key=True)
    time = Column(Float)
    edl_id = Column(Integer, ForeignKey('edl.id'))

class ClipName(Base):
    __tablename__ = 'clip_name'
    id = Column(Integer, primary_key=True)
    name = Column(String)
    edl_id = Column(Integer, ForeignKey('edl.id'))


def init_db(db_file):
    engine = create_engine(f'sqlite:///{db_file}')
    Base.metadata.create_all(engine)
    return engine

def add_edl(engine, edl):
    Session = sessionmaker(bind=engine)
    session = Session()
    session.add(edl)
    session.commit()
class SRT(Base):
    __tablename__ = 'srt'
    id = Column(Integer, primary_key=True)
    video_id = Column(Integer, ForeignKey('video.id'))
    audio_id = Column(Integer, ForeignKey('audio.id'))
    subtitles = Column(String)

    video = relationship('Video', backref=backref('srts', uselist=True))
    audio = relationship('Audio', backref=backref('srts', uselist=True))

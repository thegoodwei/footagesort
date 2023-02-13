from sqlalchemy import create_engine, Column, Integer, String, Float, ForeignKey
from sqlalchemy.orm import relationship, sessionmaker
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()

class EDL(Base):
    __tablename__ = 'edl'
    id = Column(Integer, primary_key=True)
    project_name = Column(String)
    video_files = Column(String)
    audio_files = Column(String)
    timecode = Column(String)
    edits = Column(String)
    play_times = Column(String)
    clip_names = Column(String)
    fps = Column(Float)
    codec = Column(String)
    resolution = Column(String)
    aspect_ratio = Column(String)
    total_clips = Column(Integer)
    total_runtime = Column(Float)
    subtitles_srt = Column(String)
    metadata = Column(String)

class VideoFiles(Base):
    __tablename__ = 'video_files'
    id = Column(Integer, primary_key=True)
    file_path = Column(String)
    edl_id = Column(Integer, ForeignKey('edl.id'))
    edl = relationship("EDL", back_populates="video_files")

class AudioFiles(Base):
    __tablename__ = 'audio_files'
    id = Column(Integer, primary_key=True)
    file_path = Column(String)
    edl_id = Column(Integer, ForeignKey('edl.id'))
    edl = relationship("EDL", back_populates="audio_files")

EDL.video_files = relationship("VideoFiles", order_by=VideoFiles.id, back_populates="edl")
EDL.audio_files = relationship("AudioFiles", order_by=AudioFiles.id, back_populates="edl")

engine = create_engine('sqlite:///edl.db')
Base.metadata.create_all(engine)

Session = sessionmaker(bind=engine)
session = Session()

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

def init_db(db_file):
    engine = create_engine(f'sqlite:///{db_file}')
    Base.metadata.create_all(engine)
    return engine

def add_edl(engine, edl):
    Session = sessionmaker(bind=engine)
    session = Session()
    session.add(edl)
    session.commit()

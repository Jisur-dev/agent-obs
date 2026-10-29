from sqlalchemy import create_engine  
from sqlalchemy.orm import Session, sessionmaker
from .base import Base

DATABASE_URL="sqlite:///agent_obs.db"   
engine = create_engine(DATABASE_URL)


SessionLocal = sessionmaker(bind=engine)

def get_session() -> Session:
    
    return SessionLocal()


def create_tables() -> None:
    Base.metadata.create_all(bind=engine)
   
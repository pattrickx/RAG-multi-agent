# Table: users
# Columns: user_name (string) unique primary key, password (string)
# Table: chats
# Columns: chat_id (integer) primary key, user_name (string) FK -> users(user_name), chat_name (string)
# Table: groups
# Columns: group_id (integer) primary key, group_name (string), user_name (string) FK -> users(user_name)
# Table: files
# Columns: file_id (integer) primary key, file_hash (string) unique per group, file_name (string),
#   file_status (string) [unprocessed, processed, in_progress, error], user_name (string) FK -> users(user_name),
#   group_id (integer) FK -> groups(group_id)
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Column, String, Integer, ForeignKey, UniqueConstraint
from sqlalchemy.orm import relationship
from sqlalchemy.ext.asyncio import AsyncEngine

Base = declarative_base()

class User(Base):
    __tablename__ = 'users'
    user_name = Column(String, primary_key=True, unique=True)
    password = Column(String)
    chats = relationship("Chat", back_populates="user")
    groups = relationship("Group", back_populates="user")
    files = relationship("File", back_populates="user")

class Chat(Base):
    __tablename__ = 'chats'
    chat_id = Column(Integer, primary_key=True)
    user_name = Column(String, ForeignKey('users.user_name'))
    chat_name = Column(String)
    user = relationship("User", back_populates="chats")

class Group(Base):
    __tablename__ = 'groups'
    group_id = Column(Integer, primary_key=True)
    group_name = Column(String)
    user_name_owner = Column(String, ForeignKey('users.user_name'))
    user = relationship("User", back_populates="groups")
class GroupMember(Base):
    __tablename__ = 'group_members'
    id = Column(Integer, primary_key=True)
    group_id = Column(Integer, ForeignKey('groups.group_id'))
    user_name_member = Column(String, ForeignKey('users.user_name'))
    group = relationship("Group")
    user = relationship("User")
class ChatsGroups(Base):
    __tablename__ = 'chats_groups'
    id = Column(Integer, primary_key=True)
    chat_id = Column(Integer, ForeignKey('chats.chat_id'))
    group_id = Column(Integer, ForeignKey('groups.group_id'))
    chat = relationship("Chat")
    group = relationship("Group")

class File(Base):   
    __tablename__ = 'files'
    file_id = Column(Integer, primary_key=True)
    file_hash = Column(String, unique=True)
    file_name = Column(String)
    file_status = Column(String)  # unprocessed, processed, in_progress, error
    total_pages = Column(Integer, default=-1)  # updated when file is processed; tracks progress and enables resume from last loaded page
    loaded_pages = Column(Integer, default=0)
    user_name = Column(String, ForeignKey('users.user_name'))
    group_id = Column(Integer, ForeignKey('groups.group_id'))
    user = relationship("User", back_populates="files")
    group = relationship("Group")
    __table_args__ = (UniqueConstraint('file_hash', 'group_id', name='_file_hash_group_uc'),)




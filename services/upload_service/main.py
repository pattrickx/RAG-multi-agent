"""
Upload Service — FastAPI app for file, user and group management.

Endpoints:
    /users/*    — user CRUD
    /chats/*    — chat CRUD
    /groups/*   — group CRUD
    /files/*    — file CRUD (metadata)
    /s3/*       — S3 operations (upload, download, list, delete)
"""

import io
import mimetypes
import os
import secrets
from contextlib import asynccontextmanager
from typing import Optional

import httpx
from botocore.exceptions import NoCredentialsError, PartialCredentialsError
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, File as FastAPIFile, HTTPException, UploadFile, status
from fastapi.responses import StreamingResponse
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from pydantic import BaseModel, HttpUrl
from sqlalchemy import delete, update
from sqlalchemy.exc import DBAPIError
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.future import select

from orm import Base, Chat, ChatsGroups, File as FileModel, Group, GroupMember, User
from s3_client import (
    create_bucket,
    delete_bucket,
    delete_file as s3_delete_file,
    list_bucket_contents,
    s3,
    upload_file,
)

load_dotenv()


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DATABASE_URL = os.environ.get("DATABASE_URL")
DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 12010


# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

security = HTTPBasic()
engine = create_async_engine(DATABASE_URL, echo=True, future=True)


async def _create_tables() -> None:
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.started = True
    await _create_tables()
    yield
    await engine.dispose()


app = FastAPI(lifespan=lifespan)


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class DownloadRequest(BaseModel):
    url: HttpUrl
    filename: Optional[str] = None


class UserCreate(BaseModel):
    user_name: str
    password: str


class ChatCreate(BaseModel):
    user_name: str
    chat_name: str


class GroupCreate(BaseModel):
    user_name: str
    group_name: str


# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------

async def require_login(
    credentials: HTTPBasicCredentials = Depends(security),
) -> User:
    """Authenticates user via HTTP Basic Auth."""
    async with AsyncSession(engine) as session:
        result = await session.execute(
            select(User).where(User.user_name == credentials.username)
        )
        user = result.scalars().first()

        if not user or not secrets.compare_digest(user.password, credentials.password):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid credentials",
                headers={"WWW-Authenticate": "Basic"},
            )
        return user


# ---------------------------------------------------------------------------
# User endpoints
# ---------------------------------------------------------------------------

@app.post("/users/")
async def create_user(user: UserCreate) -> dict:
    """Creates new user with default group, chat and membership."""
    async with AsyncSession(engine) as session:
        try:
            new_user = User(user_name=user.user_name, password=user.password)
            session.add(new_user)

            default_group = Group(user_name_owner=user.user_name, group_name="default")
            session.add(default_group)

            default_group_member = GroupMember(
                user_name_member=user.user_name, group_id=default_group.group_id
            )
            session.add(default_group_member)

            default_chat = Chat(user_name=user.user_name, chat_name="default")
            session.add(default_chat)

            default_chat_group = ChatsGroups(
                chat_id=default_chat.chat_id, group_id=default_group.group_id
            )
            session.add(default_chat_group)

            await session.commit()
            return {"message": "User created successfully"}
        except DBAPIError as e:
            await session.rollback()
            if "No space left on device" in str(e):
                raise HTTPException(
                    status_code=507, detail="Insufficient storage on database server"
                )
            raise HTTPException(status_code=500, detail="Database error while creating user")


@app.get("/users/{user_name}/info/")
async def get_user_info(
    user_name: str, current_user: User = Depends(require_login)
) -> dict:
    """Returns user info (groups and chats)."""
    if current_user.user_name != user_name:
        raise HTTPException(status_code=403, detail="Not allowed")

    async with AsyncSession(engine) as session:
        result = await session.execute(select(User).where(User.user_name == user_name))
        user = result.scalars().first()
        if not user:
            raise HTTPException(status_code=404, detail="User not found")

        groups = (await session.execute(
            select(Group).where(Group.user_name_owner == user_name)
        )).scalars().all()

        chats = (await session.execute(
            select(Chat).where(Chat.user_name == user_name)
        )).scalars().all()

        return {
            "user_name": user.user_name,
            "groups": [{"group_id": g.group_id, "group_name": g.group_name} for g in groups],
            "chats": [{"chat_id": c.chat_id, "chat_name": c.chat_name} for c in chats],
        }


@app.delete("/users/{user_name}/")
async def delete_user(
    user_name: str, current_user: User = Depends(require_login)
) -> dict:
    """Deletes user and all associated resources."""
    if current_user.user_name != user_name:
        raise HTTPException(status_code=403, detail="Not allowed")

    async with AsyncSession(engine) as session:
        result = await session.execute(select(User).where(User.user_name == user_name))
        user = result.scalars().first()
        if not user:
            raise HTTPException(status_code=404, detail="User not found")

        groups = (await session.execute(
            select(Group).where(Group.user_name_owner == user_name)
        )).scalars().all()

        for group in groups:
            await session.execute(
                update(FileModel)
                .where(FileModel.group_id == group.group_id)
                .values(file_status="error")
            )
            await session.execute(
                delete(GroupMember).where(GroupMember.group_id == group.group_id)
            )
            await session.delete(group)

        await session.execute(delete(Chat).where(Chat.user_name == user_name))
        await session.delete(user)
        await session.commit()
        return {"message": "User deleted successfully"}


# ---------------------------------------------------------------------------
# Chat endpoints
# ---------------------------------------------------------------------------

@app.post("/chats/")
async def create_chat(
    chat: ChatCreate, current_user: User = Depends(require_login)
) -> dict:
    """Creates new chat with associated group."""
    if current_user.user_name != chat.user_name:
        raise HTTPException(status_code=403, detail="Not allowed")

    async with AsyncSession(engine) as session:
        result = await session.execute(
            select(User).where(User.user_name == chat.user_name)
        )
        if not result.scalars().first():
            raise HTTPException(status_code=404, detail="User not found")

        new_chat = Chat(user_name=chat.user_name, chat_name=chat.chat_name)
        session.add(new_chat)

        new_group = Group(user_name_owner=chat.user_name, group_name=chat.chat_name)
        session.add(new_group)

        session.add(GroupMember(
            user_name_member=chat.user_name, group_name=chat.chat_name
        ))
        session.add(ChatsGroups(chat_id=new_chat.chat_id, group_id=new_group.group_id))

        await session.commit()
        return {"message": "Chat created successfully"}


@app.delete("/chats/{chat_id}/")
async def delete_chat(
    chat_id: int, current_user: User = Depends(require_login)
) -> dict:
    """Deletes chat and associated group."""
    async with AsyncSession(engine) as session:
        result = await session.execute(select(Chat).where(Chat.chat_id == chat_id))
        chat = result.scalars().first()
        if not chat:
            raise HTTPException(status_code=404, detail="Chat not found")
        if chat.user_name != current_user.user_name:
            raise HTTPException(status_code=403, detail="Not allowed")

        group = (await session.execute(
            select(Group).where(
                Group.group_name == chat.chat_name,
                Group.user_name_owner == chat.user_name,
            )
        )).scalars().first()

        if group:
            await session.execute(
                update(FileModel)
                .where(FileModel.group_id == group.group_id)
                .values(file_status="error")
            )
            await session.execute(
                delete(GroupMember).where(GroupMember.group_id == group.group_id)
            )
            await session.delete(group)
            await session.execute(
                delete(ChatsGroups).where(ChatsGroups.chat_id == chat.chat_id)
            )

        await session.delete(chat)
        await session.commit()
        return {"message": "Chat deleted successfully"}


# ---------------------------------------------------------------------------
# Group endpoints
# ---------------------------------------------------------------------------

@app.post("/groups/")
async def create_group(
    group: GroupCreate, current_user: User = Depends(require_login)
) -> dict:
    """Creates new group."""
    if current_user.user_name != group.user_name:
        raise HTTPException(status_code=403, detail="Not allowed")

    async with AsyncSession(engine) as session:
        result = await session.execute(
            select(User).where(User.user_name == group.user_name)
        )
        if not result.scalars().first():
            raise HTTPException(status_code=404, detail="User not found")

        session.add(Group(
            user_name_owner=group.user_name, group_name=group.group_name
        ))
        await session.commit()
        return {"message": "Group created successfully"}


@app.delete("/groups/{group_id}/")
async def delete_group(
    group_id: int, current_user: User = Depends(require_login)
) -> dict:
    """Deletes group and marks files as error."""
    async with AsyncSession(engine) as session:
        result = await session.execute(select(Group).where(Group.group_id == group_id))
        group = result.scalars().first()
        if not group:
            raise HTTPException(status_code=404, detail="Group not found")
        if group.user_name_owner != current_user.user_name:
            raise HTTPException(status_code=403, detail="Not allowed")

        await session.execute(
            update(FileModel)
            .where(FileModel.group_id == group_id)
            .values(file_status="error")
        )
        await session.execute(
            delete(GroupMember).where(GroupMember.group_id == group_id)
        )
        await session.delete(group)
        await session.commit()
        return {"message": "Group deleted successfully"}


# ---------------------------------------------------------------------------
# File endpoints
# ---------------------------------------------------------------------------

@app.get("/files/")
async def get_files_by_status(
    file_status: str, current_user: User = Depends(require_login)
):
    """Lists files by status for the logged-in user."""
    async with AsyncSession(engine) as session:
        result = await session.execute(
            select(FileModel).where(
                FileModel.file_status == file_status,
                FileModel.user_name == current_user.user_name,
            )
        )
        return result.scalars().all()


@app.put("/files/{file_id}/status/")
async def update_file_status(
    file_id: int,
    file_status: str,
    loaded_pages: int = -1,
    total_pages: int = -1,
    current_user: User = Depends(require_login),
) -> dict:
    """Updates file status."""
    async with AsyncSession(engine) as session:
        result = await session.execute(
            select(FileModel).where(FileModel.file_id == file_id)
        )
        file = result.scalars().first()
        if not file:
            raise HTTPException(status_code=404, detail="File not found")
        if file.user_name != current_user.user_name:
            raise HTTPException(status_code=403, detail="Not allowed")

        file.file_status = file_status
        if loaded_pages >= 0:
            file.loaded_pages = loaded_pages
        if total_pages >= 0:
            file.total_pages = total_pages
        await session.commit()
        return {"message": "File status updated successfully"}


@app.delete("/files/{file_id}/")
async def delete_file_record(
    file_id: int, current_user: User = Depends(require_login)
) -> dict:
    """Deletes file from S3 and database."""
    async with AsyncSession(engine) as session:
        result = await session.execute(
            select(FileModel).where(FileModel.file_id == file_id)
        )
        file = result.scalars().first()
        if not file:
            raise HTTPException(status_code=404, detail="File not found")
        if file.user_name != current_user.user_name:
            raise HTTPException(status_code=403, detail="Not allowed")

        bucket_name = f"{file.user_name}_{file.group_id}"
        s3_delete_file(bucket_name, file.file_name)

        await session.delete(file)
        await session.commit()
        return {"message": "File deleted successfully"}


# ---------------------------------------------------------------------------
# S3 endpoints
# ---------------------------------------------------------------------------

@app.post("/s3/upload/")
async def upload_to_s3(
    bucket_name: str,
    files: list[UploadFile] = FastAPIFile(...),
    current_user: User = Depends(require_login),
) -> dict:
    """Uploads files to S3."""
    if not files:
        raise HTTPException(status_code=400, detail="At least one file is required")

    try:
        async with AsyncSession(engine) as session:
            user = (await session.execute(
                select(User).where(User.user_name == current_user.user_name)
            )).scalars().first()
            if not user:
                raise HTTPException(status_code=404, detail="User not found")

            group = (await session.execute(
                select(Group).where(
                    Group.group_name == bucket_name,
                    Group.user_name_owner == current_user.user_name,
                )
            )).scalars().first()
            if not group:
                raise HTTPException(status_code=404, detail="Group not found")

            uploaded_files = []
            for file in files:
                file_location = f"/tmp/{current_user.user_name}_{file.filename}"
                try:
                    with open(file_location, "wb") as buffer:
                        buffer.write(await file.read())

                    upload_file(bucket_name, file_location, file.filename)

                    session.add(FileModel(
                        file_hash=file.filename,
                        file_name=file.filename,
                        file_status="unprocessed",
                        user_name=current_user.user_name,
                        group_id=group.group_id,
                        loaded_pages=0,
                        total_pages=0,
                    ))
                    uploaded_files.append(file.filename)
                finally:
                    if os.path.exists(file_location):
                        os.remove(file_location)

            await session.commit()

        return {
            "message": f"{len(uploaded_files)} file(s) uploaded to '{bucket_name}'.",
            "files": uploaded_files,
        }
    except NoCredentialsError:
        raise HTTPException(status_code=401, detail="AWS credentials not found")
    except PartialCredentialsError:
        raise HTTPException(status_code=401, detail="Incomplete AWS credentials")


@app.get("/s3/list/")
async def list_s3_bucket(
    bucket_name: str, current_user: User = Depends(require_login)
) -> dict:
    """Lists bucket contents."""
    try:
        contents = list_bucket_contents(bucket_name)
        return {"bucket": bucket_name, "contents": contents}
    except s3.exceptions.NoSuchBucket:
        raise HTTPException(status_code=404, detail=f"Bucket '{bucket_name}' does not exist")
    except NoCredentialsError:
        raise HTTPException(status_code=401, detail="AWS credentials not found")
    except PartialCredentialsError:
        raise HTTPException(status_code=401, detail="Incomplete AWS credentials")


@app.delete("/s3/delete_file/")
async def delete_file_from_s3(
    bucket_name: str, key: str, current_user: User = Depends(require_login)
) -> dict:
    """Deletes file from bucket."""
    try:
        s3_delete_file(bucket_name, key)
        return {"message": f"File '{key}' deleted from '{bucket_name}'."}
    except s3.exceptions.NoSuchKey:
        raise HTTPException(status_code=404, detail=f"File '{key}' not found in '{bucket_name}'")
    except s3.exceptions.NoSuchBucket:
        raise HTTPException(status_code=404, detail=f"Bucket '{bucket_name}' does not exist")
    except NoCredentialsError:
        raise HTTPException(status_code=401, detail="AWS credentials not found")
    except PartialCredentialsError:
        raise HTTPException(status_code=401, detail="Incomplete AWS credentials")


@app.delete("/s3/delete_bucket/")
async def delete_s3_bucket(
    bucket_name: str, current_user: User = Depends(require_login)
) -> dict:
    """Deletes entire bucket."""
    try:
        delete_bucket(bucket_name)
        return {"message": f"Bucket '{bucket_name}' deleted."}
    except s3.exceptions.NoSuchBucket:
        raise HTTPException(status_code=404, detail=f"Bucket '{bucket_name}' does not exist")
    except NoCredentialsError:
        raise HTTPException(status_code=401, detail="AWS credentials not found")
    except PartialCredentialsError:
        raise HTTPException(status_code=401, detail="Incomplete AWS credentials")


@app.get("/s3/download/")
async def download_from_s3(
    bucket_name: str, key: str, current_user: User = Depends(require_login)
):
    """Downloads file from S3."""
    try:
        s3_object = s3.get_object(Bucket=bucket_name, Key=key)
        file_stream = io.BytesIO(s3_object["Body"].read())
        mime_type = mimetypes.guess_type(key)[0] or "application/octet-stream"
        return StreamingResponse(
            file_stream,
            media_type=mime_type,
            headers={"Content-Disposition": f"attachment; filename={key}"},
        )
    except s3.exceptions.NoSuchKey:
        raise HTTPException(status_code=404, detail=f"File '{key}' not found in '{bucket_name}'")
    except s3.exceptions.NoSuchBucket:
        raise HTTPException(status_code=404, detail=f"Bucket '{bucket_name}' does not exist")
    except NoCredentialsError:
        raise HTTPException(status_code=401, detail="AWS credentials not found")
    except PartialCredentialsError:
        raise HTTPException(status_code=401, detail="Incomplete AWS credentials")


@app.get("/s3/stream/")
async def stream_from_s3(
    bucket_name: str, key: str, current_user: User = Depends(require_login)
):
    """Streams file from S3."""
    try:
        s3_object = s3.get_object(Bucket=bucket_name, Key=key)
        mime_type = mimetypes.guess_type(key)[0] or "application/octet-stream"
        return StreamingResponse(
            s3_object["Body"],
            media_type=mime_type,
            headers={"Content-Disposition": f"attachment; filename={key}"},
        )
    except s3.exceptions.NoSuchKey:
        raise HTTPException(status_code=404, detail=f"File '{key}' not found in '{bucket_name}'")
    except s3.exceptions.NoSuchBucket:
        raise HTTPException(status_code=404, detail=f"Bucket '{bucket_name}' does not exist")
    except NoCredentialsError:
        raise HTTPException(status_code=401, detail="AWS credentials not found")
    except PartialCredentialsError:
        raise HTTPException(status_code=401, detail="Incomplete AWS credentials")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=DEFAULT_HOST, port=DEFAULT_PORT)

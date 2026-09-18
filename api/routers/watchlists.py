from __future__ import annotations

from typing import Annotated, List

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from api.deps import get_db, get_current_user
from api.schemas.footiq import WatchlistCreate, WatchlistItemCreate, WatchlistItemRead, WatchlistRead
from db.models import Watchlist as WatchlistModel, WatchlistItem as WatchlistItemModel, User as UserModel

router = APIRouter()


@router.post("", response_model=WatchlistRead)
async def create_watchlist(
    payload: WatchlistCreate,
    db: Annotated[AsyncSession, Depends(get_db)],
    current_user: Annotated[UserModel, Depends(get_current_user)],
) -> WatchlistModel:
    watchlist = WatchlistModel(owner_user_id=current_user.id, **payload.model_dump())
    db.add(watchlist)
    await db.flush()
    return watchlist


@router.get("", response_model=List[WatchlistRead])
async def list_watchlists(
    db: Annotated[AsyncSession, Depends(get_db)],
    current_user: Annotated[UserModel, Depends(get_current_user)],
) -> List[WatchlistModel]:
    result = await db.execute(select(WatchlistModel).where(WatchlistModel.owner_user_id == current_user.id))
    return result.scalars().all()


@router.post("/items", response_model=WatchlistItemRead)
async def add_watchlist_item(
    payload: WatchlistItemCreate,
    db: Annotated[AsyncSession, Depends(get_db)],
    current_user: Annotated[UserModel, Depends(get_current_user)],
) -> WatchlistItemModel:
    item = WatchlistItemModel(**payload.model_dump())
    db.add(item)
    await db.flush()
    return item

from __future__ import annotations

from typing import Annotated, List

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from api.deps import get_db, get_current_user
from api.schemas.footiq import OfferCreate, OfferRead
from db.models import Offer as OfferModel, User as UserModel

router = APIRouter()


@router.post("", response_model=OfferRead)
async def create_offer(
    payload: OfferCreate,
    db: Annotated[AsyncSession, Depends(get_db)],
    current_user: Annotated[UserModel, Depends(get_current_user)],
) -> OfferModel:
    offer = OfferModel(sender_user_id=current_user.id, **payload.model_dump())
    db.add(offer)
    await db.flush()
    return offer


@router.get("", response_model=List[OfferRead])
async def list_offers(
    db: Annotated[AsyncSession, Depends(get_db)],
    current_user: Annotated[UserModel, Depends(get_current_user)],
) -> List[OfferModel]:
    result = await db.execute(select(OfferModel))
    return result.scalars().all()

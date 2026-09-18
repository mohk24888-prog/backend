from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from api.deps import get_db, get_current_user
from api.schemas.footiq import PhysicalMetricRead, TacticalMetricRead, TechnicalMetricRead
from db.models import PhysicalMetric as PhysicalMetricModel, TacticalMetric as TacticalMetricModel, TechnicalMetric as TechnicalMetricModel, Analysis as AnalysisModel, User as UserModel

router = APIRouter()


@router.get("/physical/{analysis_id}", response_model=PhysicalMetricRead)
async def get_physical_metrics(
    analysis_id: str,
    db: Annotated[AsyncSession, Depends(get_db)],
    current_user: Annotated[UserModel, Depends(get_current_user)],
) -> PhysicalMetricModel:
    result = await db.execute(select(PhysicalMetricModel).where(PhysicalMetricModel.analysis_id == analysis_id))
    metric = result.scalar_one_or_none()
    if not metric:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Metrics not found")
    return metric


@router.get("/technical/{analysis_id}", response_model=TechnicalMetricRead)
async def get_technical_metrics(
    analysis_id: str,
    db: Annotated[AsyncSession, Depends(get_db)],
    current_user: Annotated[UserModel, Depends(get_current_user)],
) -> TechnicalMetricModel:
    result = await db.execute(select(TechnicalMetricModel).where(TechnicalMetricModel.analysis_id == analysis_id))
    metric = result.scalar_one_or_none()
    if not metric:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Metrics not found")
    return metric


@router.get("/tactical/{analysis_id}", response_model=TacticalMetricRead)
async def get_tactical_metrics(
    analysis_id: str,
    db: Annotated[AsyncSession, Depends(get_db)],
    current_user: Annotated[UserModel, Depends(get_current_user)],
) -> TacticalMetricModel:
    result = await db.execute(select(TacticalMetricModel).where(TacticalMetricModel.analysis_id == analysis_id))
    metric = result.scalar_one_or_none()
    if not metric:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Metrics not found")
    return metric

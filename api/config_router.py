from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel
from typing import Dict, Any, List
import json
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update

from core.database.engine import get_db_dependency
from core.database.models import SystemConfig, ConfigAudit

router = APIRouter(prefix="/api/config", tags=["Configuration"])

class ConfigUpdateModel(BaseModel):
    key: str
    value: Dict[str, Any]

@router.get("/")
async def get_all_config(db: AsyncSession = Depends(get_db_dependency)):
    result = await db.execute(select(SystemConfig))
    configs = result.scalars().all()
    return {c.key: json.loads(c.value) for c in configs}

@router.get("/{key}")
async def get_config(key: str, db: AsyncSession = Depends(get_db_dependency)):
    result = await db.execute(select(SystemConfig).where(SystemConfig.key == key))
    config = result.scalars().first()
    if not config:
        return {}
    return json.loads(config.value)

@router.post("/{key}")
async def update_config(key: str, payload: Dict[str, Any], db: AsyncSession = Depends(get_db_dependency)):
    # Basic validation
    if key == "risk":
        max_daily_loss = payload.get("max_daily_loss", 2.0)
        risk_per_trade = payload.get("risk_per_trade", 1.0)
        max_dd = payload.get("max_dd", 10.0)
        if risk_per_trade > max_daily_loss:
            raise HTTPException(status_code=400, detail="Risk per trade cannot exceed max daily loss.")
        if max_daily_loss > max_dd:
            raise HTTPException(status_code=400, detail="Max daily loss cannot exceed max drawdown.")

    result = await db.execute(select(SystemConfig).where(SystemConfig.key == key))
    config = result.scalars().first()
    
    new_value_str = json.dumps(payload)
    
    if config:
        old_value_str = config.value
        config.value = new_value_str
        config.version += 1
    else:
        old_value_str = None
        config = SystemConfig(key=key, value=new_value_str, version=1)
        db.add(config)
        
    audit = ConfigAudit(
        key=key,
        old_value=old_value_str,
        new_value=new_value_str,
        changed_by="dashboard"
    )
    db.add(audit)
    
    try:
        await db.commit()
    except Exception as e:
        await db.rollback()
        raise HTTPException(status_code=500, detail="Failed to save configuration.")
        
    return {"status": "success", "key": key, "version": config.version}

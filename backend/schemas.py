from typing import Optional, Any
from pydantic import BaseModel

class Transaction(BaseModel):
    transaction_id: Any
    timestamp: str
    amount: float
    is_fraud: int
    score: float
    action: str
    ground_truth: int
    ProductCD: Optional[str] = "U"
    device_vendor: Optional[str] = ""
    dist1: Optional[float] = 0
    card_email_combo_fraud_rate: Optional[float] = 0
    
    class Config:
        extra = "allow"

class StatsResponse(BaseModel):
    total_processed: int
    fraud_detected: int
    legit_transactions: int
    fraud_rate: float
    queue_depth: int
    updated_at: str
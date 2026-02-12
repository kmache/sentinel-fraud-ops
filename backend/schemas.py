from typing import Optional, Any
from pydantic import BaseModel, Field

class Transaction(BaseModel):
    transaction_id: Any
    timestamp: str
    
    amount: float = Field(alias="TransactionAmt") 
    
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
        populate_by_name = True

class StatsResponse(BaseModel):
   
    precision: float
    recall: float
    fpr_insult_rate: float
    auc: float
    fraud_rate: float
    f1_score: float            
    
    fraud_stopped_val: float
    fraud_missed_val: float
    false_positive_loss: float
    net_savings: float
    
    total_processed: int
    queue_depth: int
    threshold: float
    updated_at: str
    total_lifetime_count: int
    live_latency_ms: float    
from typing import Optional, Any
from pydantic import BaseModel, ConfigDict, Field


class Transaction(BaseModel):
    transaction_id: Any
    timestamp: str
    
    amount: float = Field(alias="TransactionAmt") 
    
    is_fraud: int
    score: float = Field(ge=0.0, le=1.0)
    action: str
    ground_truth: int
    ProductCD: Optional[str] = "U"
    device_vendor: Optional[str] = ""
    dist1: Optional[float] = 0
    card_email_combo_fraud_rate: Optional[float] = 0
    
    model_config = ConfigDict(extra="allow", populate_by_name=True)

class StatsResponse(BaseModel):
   
    precision: float = Field(ge=0.0, le=1.0)
    recall: float = Field(ge=0.0, le=1.0)
    fpr_insult_rate: float = Field(ge=0.0, le=1.0)
    auc: float = Field(ge=0.0, le=1.0)
    fraud_rate: float
    f1_score: float = Field(ge=0.0, le=1.0)
    
    fraud_stopped_val: float
    fraud_missed_val: float
    false_positive_loss: float
    net_savings: float
    
    total_processed: int = Field(ge=0)
    queue_depth: int = Field(ge=0)
    threshold: float = Field(ge=0.0, le=1.0)
    updated_at: str
    total_lifetime_count: int = Field(ge=0)
    live_latency_ms: float = Field(ge=0.0)
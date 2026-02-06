import numpy as np
from sklearn.metrics import (
    roc_auc_score, roc_curve, precision_recall_curve, 
    average_precision_score, confusion_matrix
)
from typing import Dict, Any, List, Union

class SentinelEvaluator:
    """
    Advanced Fraud Evaluator: Implements Precision-Recall optimization, 
    Financial Cost Minimization, and Tiered Response Strategies.
    """
    def __init__(self, y_true: Union[np.ndarray, List], y_prob: Union[np.ndarray, List], amounts: Union[np.ndarray, List] = None):
        self.y_true = np.array(y_true)
        self.y_prob = np.array(y_prob)
        
        # In fraud, amounts are critical for cost-based optimization
        if amounts is None:
            print("⚠️ Warning: Amounts not provided, defaulting to 0 for all transactions")
            self.amounts = np.zeros_like(self.y_true)
        else:
            self.amounts = np.array(amounts)

    def get_auc(self) -> float:
        """Calculates ROC-AUC. Handles edge case where only one class is present."""
        if len(np.unique(self.y_true)) < 2:
            return 0.5 
        return roc_auc_score(self.y_true, self.y_prob)

    def optimize_threshold_capped_fpr(self, max_fpr: float = 0.01) -> float:
        """Wrapper for backward compatibility."""
        return self.find_best_threshold(method='friction', max_fpr=max_fpr)

    def get_core_metrics(self) -> Dict[str, float]:
        """Returns the vital signs of the fraud model."""
        return {
            "roc_auc": self.get_auc(),
            "pr_auc": average_precision_score(self.y_true, self.y_prob),
        }

    def find_best_threshold(self, method: str = 'cost', **kwargs) -> float:
        """
        Finds optimal threshold based on different business goals.
        Methods: 'cost', 'fbeta', 'friction'
        """
        if len(self.y_true) == 0: return 0.5
        
        if method == 'friction':
            max_fpr = kwargs.get('max_fpr', 0.01)
            fpr, tpr, thresholds = roc_curve(self.y_true, self.y_prob)
            
            valid_indices = np.where(fpr <= max_fpr)[0]
            if len(valid_indices) == 0: return 0.5
            idx = valid_indices[-1]
            return float(thresholds[idx])

        elif method == 'fbeta':
            beta = kwargs.get('beta', 2.0)
            precision, recall, thresholds = precision_recall_curve(self.y_true, self.y_prob)
            
            numerator = (1 + beta**2) * (precision * recall)
            denominator = (beta**2 * precision) + recall
            
            with np.errstate(divide='ignore', invalid='ignore'):
                fscores = np.divide(numerator, denominator)
                fscores[np.isnan(fscores)] = 0.0
            
            best_idx = np.argmax(fscores)
            return float(thresholds[min(best_idx, len(thresholds)-1)])

        elif method == 'cost':
            cb_fee = kwargs.get('cb_fee', 25.0)
            support_cost = kwargs.get('support_cost', 15.0)
            churn_factor = kwargs.get('churn_factor', 0.1) 

            best_threshold = 0.5
            min_loss = float('inf')

            candidates = np.linspace(0.01, 0.99, 100)
            
            for t in candidates:
                preds = (self.y_prob >= t).astype(int)
                fn_mask = (self.y_true == 1) & (preds == 0)
                fp_mask = (self.y_true == 0) & (preds == 1)
                
                fn_loss = self.amounts[fn_mask].sum() + (fn_mask.sum() * cb_fee)
                
                fp_loss = (fp_mask.sum() * support_cost) + (self.amounts[fp_mask].sum() * churn_factor)
                
                total_loss = fn_loss + fp_loss
                
                if total_loss < min_loss:
                    min_loss = total_loss
                    best_threshold = t
            
            return float(best_threshold)
        
        return 0.5
            

    def get_tiered_strategy(self, soft_threshold: float, hard_threshold: float) -> np.ndarray:
        """
        Implements the Tiered Strategy.
        Vectorized for performance (approx 100x faster than loop).
        """
        conditions = [
            self.y_prob < soft_threshold,
            (self.y_prob >= soft_threshold) & (self.y_prob < hard_threshold),
            self.y_prob >= hard_threshold
        ]
        choices = ['APPROVE', 'REVIEW', 'BLOCK']
        
        # Default to BLOCK if edge cases arise
        return np.select(conditions, choices, default='BLOCK')

    def report_business_impact(self, threshold: float, **kwargs) -> Dict[str, Any]:
        """Detailed P&L report for a specific threshold."""
        # Allow overriding costs for reporting, or use defaults matching optimization
        cb_fee = kwargs.get('cb_fee', 25.0)
        support_cost = kwargs.get('support_cost', 15.0)
        churn_factor = kwargs.get('churn_factor', 0.1)

        preds = (self.y_prob >= threshold).astype(int)
        cm = confusion_matrix(self.y_true, preds)
        tn, fp, fn, tp = cm.ravel()

        mask_tp = (self.y_true == 1) & (preds == 1)
        mask_fn = (self.y_true == 1) & (preds == 0)
        mask_fp = (self.y_true == 0) & (preds == 1)

        # 1. Money we saved (Caught Fraud + Saved Chargeback Fees)
        fraud_caught_amt = self.amounts[mask_tp].sum()
        gross_savings = fraud_caught_amt + (tp * cb_fee)

        # 2. Money we lost by missing fraud (Missed Fraud + CB Fees)
        fraud_missed_amt = self.amounts[mask_fn].sum()
        
        # 3. Cost of False Positives (Support + Churn)
        insult_amt = self.amounts[mask_fp].sum()
        fp_wastage = (fp * support_cost) + (insult_amt * churn_factor)

        # 4. Net Savings (Gross Savings - Operational Wastage)
        net_savings = gross_savings - fp_wastage

        return {
            "performance": {
                "precision": float(round(tp / (tp + fp + 1e-6), 4)),
                "recall": float(round(tp / (tp + fn + 1e-6), 4)),
                "fpr_insult_rate": float(round(fp / (fp + tn + 1e-6), 4)),
                "auc": float(round(self.get_auc(), 4)),
                "fraud_rate": float(round(100*np.array(self.y_true).sum()/len(self.y_true), 2))
            },
            "financials": {
                "fraud_stopped_val": float(round(fraud_caught_amt, 2)),
                "fraud_missed_val": float(round(fraud_missed_amt, 2)),
                "false_positive_loss": float(round(fp_wastage, 2)),
                "net_savings": float(round(net_savings, 2)) 
            },
            "counts": {
                "total_processed": int(len(self.y_true)),
                "tp_count": int(tp),
                "fp_count": int(fp),
                "fn_count": int(fn)
            }
        }

if __name__ == "__main__":
    print("Sentinel Evaluator Loaded")



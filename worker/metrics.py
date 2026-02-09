import os
import json
import time
import redis
import logging
import sys
from datetime import datetime

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
from sentinel.evaluation import SentinelEvaluator

# Config
REDIS_HOST = os.getenv('REDIS_HOST', 'redis')
REDIS_PORT = os.getenv('REDIS_PORT', '6379')
REDIS_PASSWORD = os.getenv('REDIS_PASSWORD', None)
REFRESH_INTERVAL = 60 

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger("GlobalMetrics")

class GlobalMetricsWorker:
    def __init__(self):
        self.r = redis.Redis(
            host=REDIS_HOST,
            port=int(REDIS_PORT),
            password=REDIS_PASSWORD,
            decode_responses=True
        )
        self.y_prob = []
        self.y_true = []
        self.amounts = []
        self.last_idx = 0 
        self.cost_params = {'cb_fee': 25.0, 'support_cost': 15.0, 'churn_factor': 0.1}

        self.current_threshold = float(self.r.get('config:threshold') or 0.10898989898989898)

        self.OPTIMIZE_EVERY = 1000
        self.last_optimized_count = 0

    def sync_data(self):
        """Chronologically syncs new data from Redis into local RAM."""
        current_len = self.r.llen('stats:hist_y_prob') 
        
        if current_len > self.last_idx:
            new_probs = self.r.lrange('stats:hist_y_prob', self.last_idx, -1)
            new_trues = self.r.lrange('stats:hist_y_true', self.last_idx, -1)
            new_amts = self.r.lrange('stats:hist_amounts', self.last_idx, -1)
 
            self.y_prob.extend([float(x) for x in new_probs])
            self.y_true.extend([int(x) for x in new_trues])
            self.amounts.extend([float(x) for x in new_amts])

            self.last_idx = current_len
            return True
        return False

    def _optimize_business_strategy(self, evaluator):
        """
        Performs the heavy-duty calculations: 
        1. Generates the full Cost Curve for the Dashboard.
        2. Finds the mathematically optimal threshold.
        """
        logger.info(f"🎯 Running Strategy Optimization...")
        
        cost_curve = evaluator.get_cost_curve(self.cost_params)
        self.r.set("stats:threshold_cost_curve", json.dumps(cost_curve))
        new_optimal_t = evaluator.find_best_threshold(method='cost', **self.cost_params)
        self.current_threshold = new_optimal_t
        self.r.set('config:threshold', self.current_threshold)
        logger.info(f"✅ Strategy Updated. New Optimal Threshold: {new_optimal_t}")

    def run(self):
        logger.info(f"📈 Global Metrics Worker Started. Current Threshold: {self.current_threshold}")
        self.sync_data()
        self.last_optimized_count = len(self.y_true)
        while True:
            try:
                start_time = time.time()
                has_new_data = self.sync_data()
                total_count = len(self.y_true)

                if has_new_data and len(self.y_true) > 2:
                    
                    evaluator = SentinelEvaluator(self.y_true, self.y_prob, self.amounts)

                    delta = total_count - self.last_optimized_count
                    if delta >= self.OPTIMIZE_EVERY:
                        self._optimize_business_strategy(evaluator)
                        self.last_optimized_count = total_count

                    full_report = evaluator.report_business_impact(threshold=self.current_threshold)
                    
                    full_report['counts']['queue_depth'] = self.r.llen('sentinel_stream')
                    full_report['meta'] = {
                        "total_lifetime_count": len(self.y_true),
                        "updated_at": datetime.now().replace(microsecond=0).isoformat(),
                        "threshold": self.current_threshold
                    }

                    self.r.set("stats:stat_bussiness_report", json.dumps(full_report))
                    
                    duration = time.time() - start_time
                    logger.info(f"✅ Full Report Updated. Count: {len(self.y_true)} | Time: {duration:.2f}s")
                
            except Exception as e:
                logger.error(f"❌ Worker Error: {e}", exc_info=True)
            time.sleep(REFRESH_INTERVAL)

if __name__ == "__main__":
    worker = GlobalMetricsWorker()
    worker.run()

    
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
REFRESH_INTERVAL = 60 

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger("GlobalMetrics")

class GlobalMetricsWorker:
    def __init__(self):
        self.r = redis.Redis(host=REDIS_HOST, decode_responses=True)
        self.y_prob = []
        self.y_true = []
        self.amounts = []
        self.last_idx = 0 

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

    def run(self):
        logger.info("📈 Global Metrics Worker (Full Report Mode) Started...")
        
        while True:
            try:
                start_time = time.time()

                has_new_data = self.sync_data()

                if has_new_data and len(self.y_true) > 2:

                    current_threshold = float(self.r.get('config:threshold') or 0.5)
                    #TODO: run SentinelEvaluator.find_best_threshold to find optimal threshold and update it in Redis   
                    evaluator = SentinelEvaluator(self.y_true, self.y_prob, self.amounts)
                    full_report = evaluator.report_business_impact(threshold=current_threshold)
                    
                    full_report['meta'] = {
                        "total_lifetime_count": len(self.y_true),
                        "updated_at": datetime.now().replace(microsecond=0).isoformat(),
                        "threshold": current_threshold
                    }

                    self.r.set("stats:stat_bussiness_report", json.dumps(full_report))
                    
                    duration = time.time() - start_time
                    logger.info(f"✅ Full Report Updated. Count: {len(self.y_true)} | Time: {duration:.2f}s")
                
            except Exception as e:
                logger.error(f"❌ Worker Error: {e}")

            time.sleep(REFRESH_INTERVAL)

if __name__ == "__main__":
    worker = GlobalMetricsWorker()
    worker.run()
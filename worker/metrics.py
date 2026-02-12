import numpy as np
import os
import json
import time
import redis
import logging
import sys
from datetime import datetime

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
from sentinel.evaluation import SentinelEvaluator
from sentinel.monitoring import calculate_psi 

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

        self.last_drift_run = 0
        self.DRIFT_INTERVAL = 60 

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
    
    def check_drift_job(self):
        """
        Calculates Data Drift (PSI) for Top 30 features.
        Runs as part of the main worker loop.
        """
        logger.info("📡 Checking for Feature Drift...")
        
        importance_json = self.r.get("stats:global_feature_importance")
        if not importance_json:
            logger.warning("⚠️ No feature importance data found in Redis. Skipping drift check.")
            return
        importance_data = json.loads(importance_json)
        top_features = sorted(importance_data, key=importance_data.get, reverse=True)[:30]

        baseline_json = self.r.get("stats:training_distribution")
        if not baseline_json:
            logger.warning("⚠️ No training baseline found in Redis. Skipping drift check.")
            return
        baseline = json.loads(baseline_json)

        drift_report = {} 

        for feature in top_features:
            if feature not in baseline:
                continue
                
            live_data_raw = self.r.lrange(f"stats:hist_feat:{feature}", 0, -1)
            
            if len(live_data_raw) < 10: 
                continue
             
            try:
                live_vals = np.array([float(x) for x in live_data_raw if x is not None])
                
                # Calculate PSI using our monitoring library
                score = calculate_psi(
                    expected_pct=baseline[feature]['expected_pct'], 
                    actual_values=live_vals, 
                    bin_edges=baseline[feature]['bin_edges']
                )
                drift_report[feature] = score
            except Exception as e:
                logger.error(f"Error calculating drift for {feature}: {e}")

        if drift_report:
            self.r.set("stats:feature_drift_report", json.dumps(drift_report))
            logger.info(f"✅ Drift Report Updated for {len(drift_report)} features.")
    
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

                now = time.time()
                if (now - self.last_drift_run) > self.DRIFT_INTERVAL:
                    self.check_drift_job()
                    self.last_drift_run = now

                if has_new_data and len(self.y_true) > 2:
                    
                    evaluator = SentinelEvaluator(self.y_true, self.y_prob, self.amounts)

                    delta = total_count - self.last_optimized_count
                    if delta >= self.OPTIMIZE_EVERY:
                        self._optimize_business_strategy(evaluator)
                        logger.info("🧪 Generating True Performance Simulation Table...")
                        sim_table = evaluator.get_simulation_table()
                        self.r.set("stats:simulation_table", json.dumps(sim_table))

                        logger.info("🎯 Computing Model Calibration (Reliability Diagram)...")
                        cal_data = evaluator.get_calibration_report()
                        self.r.set("stats:calibration_data", json.dumps(cal_data))

                        self.last_optimized_count = total_count
                        
                    full_report = evaluator.report_business_impact(threshold=self.current_threshold)
                    
                    try:
                        live_latency = float(self.r.get("stats:live_latency_ms") or 0.0)
                        queue_depth = self.r.llen('sentinel_stream')
                    except Exception:
                        live_latency = 0.0
                        queue_depth = 0

                    full_report['counts']['queue_depth'] = queue_depth
                    full_report['counts']['live_latency_ms'] = live_latency
                    full_report['meta'] = {
                        "total_lifetime_count": len(self.y_true),
                        "threshold": self.current_threshold,
                        "updated_at": datetime.now().replace(microsecond=0).isoformat(),
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

    
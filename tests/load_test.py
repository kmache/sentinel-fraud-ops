"""
Locust load test for Sentinel Gateway API.

Run:
    locust -f tests/load_test.py --host=http://localhost:8000 --headless \
        -u 100 -r 10 --run-time 60s --csv=load_results

Targets: ≥1000 TPS, p50 <20ms, p95 <50ms, p99 <100ms
"""
import os
from locust import HttpUser, task, between, events


API_KEY = os.getenv("SENTINEL_API_KEY", "")


class SentinelUser(HttpUser):
    """Simulates a dashboard client polling the API."""
    wait_time = between(0.01, 0.05)  # Aggressive polling for load test

    def on_start(self):
        self.headers = {}
        if API_KEY:
            self.headers["X-API-Key"] = API_KEY

    @task(5)
    def get_stats(self):
        self.client.get("/stats", headers=self.headers, name="/stats")

    @task(10)
    def get_recent(self):
        self.client.get("/recent?limit=50", headers=self.headers, name="/recent")

    @task(3)
    def get_alerts(self):
        self.client.get("/alerts?limit=20", headers=self.headers, name="/alerts")

    @task(2)
    def get_timeseries(self):
        self.client.get("/exec/series", headers=self.headers, name="/exec/series")

    @task(1)
    def get_threshold_curve(self):
        self.client.get("/exec/threshold-optimization", headers=self.headers, name="/exec/threshold-optimization")

    @task(1)
    def health_check(self):
        self.client.get("/health", name="/health")


@events.quitting.add_listener
def print_summary(environment, **kwargs):
    """Print latency percentiles on exit."""
    stats = environment.runner.stats
    total = stats.total
    if total.num_requests == 0:
        print("No requests completed.")
        return

    print("\n" + "=" * 60)
    print("SENTINEL LOAD TEST RESULTS")
    print("=" * 60)
    print(f"Total Requests:  {total.num_requests}")
    print(f"Failures:        {total.num_failures}")
    print(f"RPS (avg):       {total.total_rps:.1f}")
    print(f"p50 latency:     {total.get_response_time_percentile(0.50):.0f} ms")
    print(f"p95 latency:     {total.get_response_time_percentile(0.95):.0f} ms")
    print(f"p99 latency:     {total.get_response_time_percentile(0.99):.0f} ms")
    print("=" * 60)

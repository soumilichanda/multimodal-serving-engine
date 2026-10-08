# Production Operations & Telemetry Runbook

### Service Boundaries & Operational SLA
- **Peak Throughput Saturation:** 83.7 req/s under 50 asynchronous client workers
- **Latency Distribution Targets:** $p50 \le 300\text{ ms}$, $p95 \le 400\text{ ms}$
- **Micro-Batch Max Capacity ($B_{\max}$):** 16 requests
- **Coalescence Temporal Window ($\Delta t_{\max}$):** 10 ms
- **Cache Eviction Strategy:** Thread-Safe LRU with TTL bounds
- **Statistical Drift Threshold:** Two-sample Kolmogorov-Smirnov test ($\alpha = 0.05$)

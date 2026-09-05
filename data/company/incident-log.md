Aug 3 — Routing engine outage, 4h11m, 61 customers affected. Root cause: schema migration run against production during business hours. Kestrel escalated to our CEO.
Aug 12 — Latency degradation 90 min. Same subsystem. Postmortem: insufficient load testing before the Kestrel volume ramp.
Aug 19 — Data export job silently failed for 9 days before a customer (Harlow Foods) noticed. No internal alert existed.
Aug 27 — Third outage on the routing engine, 2h. The on-call engineer was the same person for all three incidents; he has flagged burnout to his manager.
SLA credits issued in August: $212k. The Kestrel contract has a termination-for-convenience clause triggered by 3 SLA breaches in any rolling 90 days. This was the third.
Platform team headcount: 6. Two open reqs unfilled for 5 months.

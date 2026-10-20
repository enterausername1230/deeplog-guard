"""locust -f src/logguard/api/locustfile.py --host http://127.0.0.1:8000"""
from locust import HttpUser, constant_throughput, task


class LogUser(HttpUser):
    # 50 users * 10 requests/s = target 500 logs/s, subject to response time.
    wait_time = constant_throughput(10)

    @task
    def submit_log(self):
        self.client.post('/logs', json={
            'raw_log': '081109 203615 148 INFO dfs.DataNode$PacketResponder: '
                       'PacketResponder 1 for block blk_38865049064139660 terminating'
        })

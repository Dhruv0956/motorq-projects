import threading
import time

import speedtest


def run_speed_test():
    tester = speedtest.Speedtest()
    tester.get_best_server()
    download = tester.download() / 1_000_000
    upload = tester.upload() / 1_000_000
    return f"Download: {download:.2f} Mbps\nUpload: {upload:.2f} Mbps"


def set_timer(seconds, callback):
    def worker():
        time.sleep(seconds)
        callback()

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()
    return f"Timer set for {seconds} seconds."

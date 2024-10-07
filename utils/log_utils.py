from utils.time_utils import get_now_time


def logger(text):
    print("\n" + "*" * 10 + " RECOMMENDATION " + "*" * 10)
    print(f"[{get_now_time()}] {text}" + "\n")

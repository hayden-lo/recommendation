import time
from datetime import datetime, timedelta
from utils.toolkit import round_up


def get_now_time(time_format="%Y-%m-%d %H:%m:%S"):
    return datetime.now().strftime(time_format)


def date_add(input_date, day_span, time_format="%Y%m%d"):
    input_datetime = datetime.strptime(input_date, time_format)
    output_datetime = input_datetime + timedelta(days=day_span)
    return datetime.strftime(output_datetime, time_format)


def seconds_elapse(start_time, decimal=2):
    return round_up(time.time() - start_time, decimal)

# Author: Hayden Lao
# Script Name: progress_bar
# Created Date: Dec 16th 2018
# Description: A progress bar for loop

import sys


def progress_bar(factor, max_factor):
    """
    Method to show progress for loop action
    :param factor: Current step, usually initialize with 0[int]
    :param max_factor: Maximum step for looping[int]
    :return: Show progress bar while looping, nothing return
    """
    percentage = round(factor * 100 / int(max_factor), 2)
    arrow_num = int(percentage * 50 / 100)
    line_num = 50 - arrow_num
    process_bar = "[" + ">" * arrow_num + "-" * line_num + "]" + "%.2f" % percentage + "%" + "\r"
    if percentage != 100:
        sys.stdout.write(process_bar)
        sys.stdout.flush()
    else:
        print(process_bar)

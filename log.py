# -*- coding: utf-8 -*-
"""
Created on Mon Feb 13 09:50:09 2023

@author: Lenovo
"""

import time
import logging
def log(e):
    logger = logging.getLogger(__name__)
    logger.setLevel(level=logging.WARNING)
    handler = logging.FileHandler('log/log-%s.txt' % time.strftime('%Y-%m-%d'))
    handler.setLevel(logging.WARNING)
    formatter = logging.Formatter('#############################################################')
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.exception(msg=e)
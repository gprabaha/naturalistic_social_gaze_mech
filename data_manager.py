#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 18 14::51:42 2024

@author: pg496
"""

import logging

class DataManager:
    def __init__(self, params):
        self.params = params
        self.setup_logger()
    
    def setup_logger(self):
        """Setup the logger for the DataManager."""
        self.logger = logging.getLogger(__name__)

        
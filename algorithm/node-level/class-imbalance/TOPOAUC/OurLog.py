# -*- coding: utf-8 -*-
import logging
import getpass
import sys
import os
from config import dirs
import datetime
class OurLog(object):
    def __init__(self, init_file=None):
        user = getpass.getuser()
        self.logger = logging.getLogger(user)
        self.level=logging.DEBUG #debug < info< warning< error< critical
        self.logger.setLevel(self.level)
        if init_file == None:
            logFile = sys.argv[1]
        else:
            logFile = init_file
        today = datetime.datetime.now()
        logFile=dirs['LOGDIR']+logFile[:-4]+'_'+str(today.year)+'_'+str(today.month)+'_'+str(today.day)+'.log'
        if not os.path.exists(dirs['LOGDIR']):
            os.makedirs(dirs['LOGDIR'])
        self.formatter = logging.Formatter('%(asctime)-12s %(levelname)-8s %(name)-10s %(message)-12s')

        self.filelogHand = logging.FileHandler(logFile, encoding="utf8")
        self.filelogHand.setFormatter(self.formatter)
        self.filelogHand.setLevel(self.level)

        self.logHandSt = logging.StreamHandler()
        self.logHandSt.setLevel(self.level)
        self.logHandSt.setFormatter(self.formatter)


    def debug(self, msg):
        self.logger.handlers.clear()
        self.logger.addHandler(self.filelogHand)
        self.logger.addHandler(self.logHandSt)
        self.logger.debug(msg)

    def info(self, msg):
        self.logger.handlers.clear()
        self.logger.addHandler(self.filelogHand)
        self.logger.addHandler(self.logHandSt)
        self.logger.info(msg)

    def warn(self, msg):
        self.logger.handlers.clear()
        self.logger.addHandler(self.filelogHand)
        self.logger.addHandler(self.logHandSt)
        self.logger.warning(msg)

    def error(self, msg):
        self.logger.handlers.clear()
        self.logger.addHandler(self.filelogHand)
        self.logger.addHandler(self.logHandSt)
        self.logger.error(msg)

    def critical(self, msg):
        self.logger.handlers.clear()
        self.logger.addHandler(self.filelogHand)
        self.logger.addHandler(self.logHandSt)
        self.logger.critical(msg)

if __name__=='__main__':
    mylog=OurLog()
    mylog.debug("I'm debug")
    mylog.info("I'm info")
    mylog.warn("I'm warning")
    mylog.error("I'm error")
    mylog.critical("I'm critical")

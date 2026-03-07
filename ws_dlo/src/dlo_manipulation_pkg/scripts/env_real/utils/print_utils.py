# printutils.py
import sys
import time

def _ts():
    return time.strftime("%Y-%m-%d %H:%M:%S")

def _log(level, msg, stream=sys.stdout):
    stream.write(f"[{_ts()}] [{level}] {msg}\n")
    stream.flush()

def loginfo(msg): _log("INFO", msg, sys.stdout)
def logwarn(msg): _log("WARN", msg, sys.stdout)
def logerr(msg):  _log("ERROR", msg, sys.stderr)

# "f" variants used in your codebase (already formatted string passed in)
def loginfof(msg): loginfo(str(msg))
def logwarnf(msg): logwarn(str(msg))
def logerrf(msg):  logerr(str(msg))

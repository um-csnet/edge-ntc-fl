# Importing the library
import psutil
import time
import sys

max = 0.0

f = open("lastcpu.txt", "a")

while True:
    try:
        getCpu = float(psutil.cpu_percent(4))
        print(getCpu)
        if getCpu > max:
            max = getCpu
            print("this is max: ", max)
            f.write(str(max) + "\n")
    except KeyboardInterrupt:
        f.close()
        sys.exit(0)
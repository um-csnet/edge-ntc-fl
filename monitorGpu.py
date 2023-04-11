from jtop import jtop
import time
import sys

max = 0.0

f = open("lastgpu.txt", "a")

with jtop() as jetson:
    # jetson.ok() will provide the proper update frequency
    while True:
        try:
            # Read tegra stats
            if jetson.ok():
                getStat = jetson.stats
                getGpu = float(getStat['GPU'])
                print(getGpu)
                if getGpu > max:
                    max = getGpu
                    print("this is max: ", max)
                    f.write(str(max) + "\n")
            else:
                time.sleep(5)
        except KeyboardInterrupt:
            f.close()
            sys.exit(0)
            
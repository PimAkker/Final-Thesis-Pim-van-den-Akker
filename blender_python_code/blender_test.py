import blenderproc as bproc

import numpy as np
import sys

print(sys.version)

bproc.init()

data = bproc.renderer.render()
bproc.writer.write_hdf5("output/", data)

print('done')
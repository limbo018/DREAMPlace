import numpy as np
import os
import sys
import subprocess

# Number of samples
n = int(sys.argv[2])

# Scale factor for flute (flute can only handle int values)
scale = 1000.0

# Number of pins per net
npin = int(sys.argv[1])

# Parameters for Beta distribution
# Mode = (alpha - 1) / (alpha + beta - 2)
alpha = 1.3
beta = 6

xSpan = 100 * np.random.beta(alpha, beta, size = n)
ySpan = 100 * np.random.beta(alpha, beta, size = n)

for i in range(0, n):
    x = np.random.uniform(low = 0.0, high = xSpan[i], size = npin)
    y = np.random.uniform(low = 0.0, high = ySpan[i], size = npin)
    #x = np.array([89860.0000,  44985.5136,  44974.3444,  44811.0018])
    #y = np.array([16190.0000,  44994.1285,  45079.8538,  45062.4261])

    fh = open("tmpPts.dat", "w")
    for j in range(0, npin):
        fh.write("%d %d\n" % (scale * x[j], scale * y[j]))
    fh.close()

    # MST WL
    ps = subprocess.Popen(('cat', 'tmpPts.dat'), stdout=subprocess.PIPE)
    output = subprocess.check_output('./flute-net; exit 0', shell=True, stdin=ps.stdout)
    mstWL = (int)(output.split()[3]) / scale
    
    # HPWL
    HPWL = np.amax(x) - np.amin(x) + np.amax(y) - np.amin(y)

    # Output the sample
    entry = "%.3f %.3f" % (HPWL, mstWL)
    for j in range(0, npin):
        entry += " %.3f %.3f" % (x[j], y[j])
    print entry
    
    #exit()

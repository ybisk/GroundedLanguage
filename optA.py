import os,sys,commands
import subprocess
from hyperopt import fmin, rand, tpe, hp, Trials, exceptions, space_eval, STATUS_FAIL, STATUS_OK

def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False

def objective(x):
    lr,decay,drop = x

    cmd = "julia ModelA-S-T-RP.jl --lrate %(lr)g --decay %(decay)g --dropout %(drop)g --epochs 100 --task 1" % locals()
    print cmd
    p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    mindevloss = float("inf")
    for line in iter(p.stdout.readline,''):
        print line,
        s = line.strip().replace("(","").replace(")","").split(",")
        if len(s) >= 4 and is_number(s[3]) and float(s[3]) < mindevloss:
            mindevloss = float(s[3])
    retval = p.wait()
    print mindevloss
    return mindevloss

space = [
    hp.loguniform('lr', -10, 0),
    hp.uniform('decay', 0, 1),
    hp.uniform('drop', 0, 1)
    ]

best = fmin(objective,
    space=space,
    algo=tpe.suggest,
    max_evals=100)

print best

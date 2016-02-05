import os,sys,commands
import subprocess
import argparse
from hyperopt import fmin, rand, tpe, hp, Trials, exceptions, space_eval, STATUS_FAIL, STATUS_OK
from numpy import genfromtxt

count = 1

def get_args():
    parser = argparse.ArgumentParser(prog="hyperoptA")
    parser.add_argument("--target", default='1', choices=['1','2','3'],
        help="which target to predict: 1:source,2:target,3:direction")

    args = vars(parser.parse_args())
    return args

def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False

def objective(x):
    lr,drop,fdrop,ldrop,nlayers,hidden,gclip = x
    global count
    logfile = "../logs/hypopt/digits/t" + args['target'] + "_" + str(count) + ".csv"
    bestfile = "../logs/hypopt/digits/t" + args['target'] + "_" + str(count) + ".jld"
    count += 1
    cmd = "julia ModelA-S-T-RP-Deep-minibatch.jl --datafiles BlockWorld/digits/Train.STRP.data BlockWorld/digits/Dev.STRP.data --lr %(lr)g --gclip %(gclip)g --dropout %(drop)g --fdropout %(fdrop)g --ldropout %(ldrop)g --nlayers %(nlayers)g --batchsize 16 --nx 83 --xvocab 443 --patience 10 --epochs 100 --hidden %(hidden)g" % locals()
    cmd += " --target " + args['target']
    cmd += " --logfile " + logfile
    cmd += " --bestfile " + bestfile
    print cmd
    sys.stdout.flush()
    p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    mindevloss = float("inf")
    #for line in iter(p.stdout.readline,''):
        #print line
        #s = line.strip().replace("(","").replace(")","").split(",")
        #if len(s) >= 4 and is_number(s[3]) and float(s[3]) < mindevloss:
        #    mindevloss = float(s[3])
    retval = p.wait()
    data = genfromtxt(logfile, delimiter=',')
    r,c = data.shape
    mindevloss = data[r-1, c-1]
    print "Besterr: ", mindevloss
    sys.stdout.flush()
    return mindevloss

args = get_args()
space = [
    hp.uniform('lr', 0.0001, 0.01),
    hp.uniform('drop', 0, 1),
    hp.uniform('fdrop', 0, 1),
    hp.uniform('ldrop', 0, 1),
    hp.choice('nlayers', [2,3,4,5]),
    hp.choice('hidden', [32, 64, 128, 256]),
    hp.uniform('gclip', 1, 15)
    ]

best = fmin(objective,
    space=space,
    algo=tpe.suggest,
    max_evals=200)

hiddens = [32, 64, 128, 256]
best['nlayers'] = best['nlayers'] + 2
best['hidden'] = hiddens[best['hidden']]
print best
sys.stdout.flush()

# Feed forward neural network which predicts Source, Target, and uses them to predict XYZ coordinates
# Input File: JSONReader/data/2016-NAACL/STxyz/*.mat

using Knet
using ArgParse
using JLD
using CUDArt

# Feed forward neural network which predicts Source and predicts XYZ final location
# Input File: JSONReader/data/2016-NAACL/Sxyz/*.mat

using Knet
using ArgParse
using JLD
using CUDArt

# Feed forward neural network which predicts Source, Target, and Relative Position independantly
# Input File: JSONReader/data/2016-NAACL/STRP/*.mat

using Knet
using ArgParse
using JLD
using CUDArt

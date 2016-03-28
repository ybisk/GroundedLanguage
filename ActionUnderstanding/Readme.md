This directory contains simple Feed-Forward and Recurrent Neural Network architectures for action taking in Blocks World.
There are three paradigms presented here:

  1. SRD -- Source, Reference, Direction
    * Input:   Utterance (vector of word IDs)
    * Output:  SoftMax of 20-D (source), 20-D (Reference), 9-D (Direction)

  2. SRxyz -- Source, Reference, (x,y,z)
    * Input:   World (60-D) & Utterance (vector of word IDs)
    * Output:  SoftMax of 20-D (source), 20-D (Reference), 3-D (x,y,z)

  3. Sxyz -- Source, (x,y,z)
    * Input:   World (60-D) & Utterance (vector of word IDs)
    * Output:  SoftMax of 20-D (source), 3-D Target (x,y,z)

File formats:  predictions first and then input.
Source [Reference / Direction / XYZ ] [World] Utterance 

Evaluation:

| Architecture |  Model   | Source | Target | RP | Mean Err | Median Err |
| ------------ |  ------- |:------:|:------:|:---:|:--------:|:----------:|
|  FFN         | SRD      |        |        |     |          |            |
|  FFN         | SRxyz    |        |        |     |          |            |
|  FFN         | Sxyz     |        |        |     |          |            |
|  RNN         | SRD      |        |        |     |          |            |
|  RNN         | SRxyz    |        |        |     |          |            |
|  RNN         | Sxyz     |        |        |     |          |            |


************
Deniz notes:
************

SRD File format:
----------------
For example in JSONReader/data/2016-NAACL/SRD/Train.mat:
Source/Reference are Int block ids (0-19).
Direction is an Int (0-9).
Utterance consists of 80 Int word ids (1-658).
word=1 is used as padding at the end for short sentences.
word=1 is also used for UNK inside a sentence?
Note that word ids are 1-based, others are 0-based.

RNN-SRD Experiments:
--------------------
Common parameters and scripts:
Dict{Symbol,Any}(:lr=>1.0,:target=>3,:savefile=>nothing,:loadfile=>nothing,:dropout=>0.5,:bestfile=>nothing,:embedding=>0,:gclip=>5.0,:nogpu=>false,:ftype=>"Float32",:hidden=>256,:epochs=>50,:decay=>0.9,:xsparse=>false,:seed=>20160427,:batchsize=>10,:datafiles=>ASCIIString["JSONReader/data/2016-NAACL/SRD/Train.mat","JSONReader/data/2016-NAACL/SRD/Dev.mat"])
for i in *.out; do echo $i; grep '^(' $i | sort -k4 -g -t, | head -1; done
for i in `myqueue | perl -lne 'print $1 if /\b(hpc\d\d\d\d)\b/'`; do echo $i; ssh $i cat $OU | grep '^(' | sort -k4 -g -t, | head -1; done

Target	Hidden	TrnLogp	DevErr	BestDevEpoch
1	16	.064	.035	18
1	32	.019	.031	6
1	64	.006	.030	14	***
1	128	.015	.034	5
1	256	.014	.033	5

Target	Hidden	TrnLogp	DevErr	BestDevEpoch
2	64	.0017	.1396	20+
2	128	.0011	.1315	18	***
2	256	.0032	.1326	12

Target	Hidden	TrnLogp	DevErr	BestDevEpoch
3	64	.0733	.3287	13
3	128	.0171	.3316	20+
3	256	.0039	.3112	23	***
3	512	.0011	.3211	29
3	1024	.0056	.3362	21

Other attempts to improve target 3 fail (over the .3112 best result).
a. Increasing dropout to 0.75. => .3187@30
b. Adding dropout to wvec (input embedding) as well as hvec (hidden). => .3182@38
@@c. Trying a two layer LSTM instead of one. => 
d. Trying embedding sizes different than hidden size. => .3223@64emb .3315@128emb .3246@512emb

@@Fixing padding in RNN-SRD4. Rerunning best models for target=1,2,3.

Sxyz File format:
-----------------
For example in JSONReader/data/2016-NAACL/Sxyz/Train.mat:
147 columns.
Source id in [0,19].
63 columns of Float xyz coordinates. -1.0 marks nonexistent blocks, but also sometimes a legitimate coordinate? Is the same block (e.g. coca-cola) always in the same columns?
Variable number of columns for word ids. min=4, max=83.
Word id in [1,658].  Again 1 is probably used for unk.  We should use 0 for padding.



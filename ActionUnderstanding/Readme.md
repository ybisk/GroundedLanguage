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

Evaluation: (dev/test)

| Architecture |  Model   | Source | Ref | Dir | Mean Err | Median Err |
| ------------ |  ------- |:------:|:------:|:---:|:--------:|:----------:|
|  FFN         | SRD      |        |        |     |          |            |
|  FFN         | SRxyz    |        |        |     |          |            |
|  FFN         | Sxyz     |        |        |     |          |            |
|  RNN         | SRD      | .0332/.0211 | .1332/.0822 | .3316/.2200 |          |            |
|  RNN         | SRxyz    |        |        |     |          |            |
|  RNN         | Sxyz     |        |        |     |          |            |
|  RNN         | Txyz     |        |        |     |          |            |


```
Deniz results:

I used .1254 for block size and the quadloss to block length
conversion is done as sqrt(2*qloss)/.1254.  Note that this is
different from the average distance.

RNN-SRD zeroone loss dev/test:
RNN-SRD/S: .0332/.0211
RNN-SRD/R: .1332/.0822
RNN-SRD/D: .3316/.2200

RNN-SRD zeroone loss dev/test on blank data:
RNN-SRD/S: .8944/.9000
RNN-SRD/R: .9278/.9097
RNN-SRD/D: .5917/.5403

RNN-STxyz quadloss dev/test:
RNN-STxyz/S: .0140/.0112  (1.10/0.98 blocks) (alt result w/o test dropout: .0146/.0128)
RNN-STxyz/T: .0412/.0343  (1.88/1.72 blocks) (alt result w/o test dropout: .0411/.0333)

FFN-STxyz quadloss dev/test:
FFN-STxyz/S: .0285/.0280  (1.57/1.55 blocks)
FFN-STxyz/T: .0954/.0799  (2.87/2.62 blocks)

RNN-STxyz quadloss dev/test on blank data:
RNN-STxyz/S: .1767/.1739  (3.90/3.87 blocks) (alt result w/o test dropout: .1840/.1815)
RNN-STxyz/T: .2056/.1980  (4.21/4.13 blocks) (alt result w/o test dropout: .2004/.1854)

FFN-STxyz quadloss dev/test on blank data:
FFN-STxyz/S: .1902/.1751  (4.05/3.88 blocks)
FFN-STxyz/T: .2317/.2201  (4.47/4.35 blocks)

FFN-SRDxyz quadloss dev/test: (using gold SRD)
FFN-SRDxyz/S: .0025/.0038
FFN-SRDxyz/T: .0235/.0268

FFN-SRDxyz quadloss dev/test: (using gold SRD on blank data)
FFN-SRDxyz/S: .0013/.0020
FFN-SRDxyz/T: .0570/.0398
```

```
************
Deniz notes:
************

Board dimensions: [-1,1]
Block dimensions: .1524

SRD File format:
----------------
For example in JSONReader/data/2016-NAACL/SRD/Train.mat:
Source/Reference are Int block ids (0-19).
Direction is an Int (0-8).
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
c. Trying a two layer LSTM instead of one. => .3211@36
d. Trying embedding sizes different than hidden size. => .3223@64emb .3315@128emb .3246@512emb

Fixing padding (using 0 instead of 1 to pad, so unk tokens are not skipped in backprop):
target=1 .0320@13
target=2 .1344@11
target=3 .3188@27

Getting rid of arbitrary sentence length restriction: (epoch,lr,trnloss,deverr,tsterr)
target1: (15,0.5904900000000002,0.004885856288433999,0.03315881326352531,0.021089077746301543)
target2: (21,0.47829690000000014,0.0003369556060988965,0.133216986620128,0.0821529745042493)
target3: (47,0.13508517176729928,0.0004211945076236206,0.33158813263525305,0.22001888574126535)

Sxyz File format:
-----------------
For example in JSONReader/data/2016-NAACL/Sxyz/Train.mat:
147 columns.
Source id in [0,19].
3 columns of target xyz.
60 columns of Float xyz coordinates. -1.0 marks nonexistent blocks, but also sometimes a legitimate coordinate? Is the same block (e.g. coca-cola) always in the same columns?
Variable number of columns for word ids. min=4, max=83.
Word id in [1,658].  Again 1 is probably used for unk.  We should use 0 for padding.
```

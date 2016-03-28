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

For example in JSONReader/data/2016-NAACL/SRD/Train.mat:
Source/Reference are Int block ids (0-19).
Direction is an Int (0-9).
Utterance consists of 80 Int word ids (1-658).
word=1 is used as padding at the end for short sentences.
word=1 is also used for UNK inside a sentence?
Note that word ids are 1-based, others are 0-based.

Evaluation:

| Architecture |  Model   | Source | Target | RP | Mean Err | Median Err |
| ------------ |  ------- |:------:|:------:|:---:|:--------:|:----------:|
|  FFN         | SRD      |        |        |     |          |            |
|  FFN         | SRxyz    |        |        |     |          |            |
|  FFN         | Sxyz     |        |        |     |          |            |
|  RNN         | SRD      |        |        |     |          |            |
|  RNN         | SRxyz    |        |        |     |          |            |
|  RNN         | Sxyz     |        |        |     |          |            |

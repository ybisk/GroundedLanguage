This directory contains simple Feed-Forward and Recurrent Neural Network architectures for action taking in Blocks World.
There are three paradigms presented here:

  1. SRD -- Source, Reference, Direction
    Input:   Utterance (vector of word IDs)
    Output:  SoftMax of 20-D (source), 20-D (Reference), 9-D (Direction)
  2. SRxyz -- Source, Reference, (x,y,z)
    Input:   World (60-D) & Utterance (vector of word IDs)
    Output:  SoftMax of 20-D (source), 20-D (Reference), 3-D (x,y,z)
  3. Sxyz -- Source, (x,y,z)
    Input:   World (60-D) & Utterance (vector of word IDs)
    Output:  SoftMax of 20-D (source), 3-D Target (x,y,z)

File formats:  predictions first and then input
Source [Referece / Direction / XYZ ] [World] Utterance 

Evaluation:

  |  Model   | Source | Target | RP | Mean Err | Median Err |
  |  ------- | ------ | ------ | -- | -------- | ---------- |

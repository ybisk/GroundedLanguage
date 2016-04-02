This directory contains simple Feed-Forward and Recurrent Neural Network architectures for action taking in Blocks World.
There are three paradigms presented here:

  1. SRD -- Source, Reference, Direction
    * Input:   Utterance (vector of word IDs)
    * Output:  SoftMax of 20-D (source), 20-D (Reference), 9-D (Direction)

  2. SRDxyz -- Source, Reference, Direction, (x,y,z)
    * Input:   Source, Reference, Direction, World (60-D)
    * Output:  3-D Source (x,y,z) and 3-D Target (x,y,z)

  3. STxyz -- Source/Target, (x,y,z)
    * Input:   World (60-D) & Utterance (vector of word IDs)
    * Output:  3-D Source (x,y,z) and 3-D Target (x,y,z)

File formats:  predictions first and then input.
[Source / Reference / Direction / sXYZ / tXYZ] [World / Utterance]

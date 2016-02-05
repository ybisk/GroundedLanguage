## Models ##
**Model A**

Text --> 3 Softmax predictions :  Source, Target, Relative Position

**Model B**

Text + World --> 3 Softmax predictions 

**Model C**

Text + World --> (x,y,z) of Source and (x,y,z) for final position

## World Representations ##
- Raw :  (x,y,z) x 20
- Linear:  Linearized 2/3D grid of positions
- Conv:  Convolutions + Pooling

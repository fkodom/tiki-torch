# DARCNET -- Deep AutoRegressive Convolutional NETwork

Last updated: 08/19/2019


## License

Copyright (c) 2019 -- Radiance Technologies, Inc.

*UNCLASSIFIED//FOUO* -- All contained data and source files are UNCLASSIFIED, but are not openly available to non-contractors.  Do not distribute outside of Radiance Technologies or their government customers.


Please see `LICENSE.txt` for full details on code usage and distribution.


## About

Contains source code for training/testing DARCNET models. These are neural networks designed for detection of PIR targets, which operate by characterizing and removing background noise.  Output from the network is typically a sparse array, and detections can be obtained by simple thresholding.


## Pre-Trained Models

`original.dict` - Original DARCNET model.  Base for all other models, although multiple improvements have since been made.

`summed-box.dict` - Utilizes an initial summed-box filter, which increased accuracy and decreased the number of convolutions needed.

`tiny.dict` - Highly optimized DARCNET model, which contains only about **700 parameters**.  

`tiny-updated.dict` - Identical to `tiny-darcnet.pt`, but has been trained on updated datasets, which contain significantly more sensor noise.  



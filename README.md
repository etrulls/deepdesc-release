# Deep Descriptors

This repository contains the code release for our 2015 ICCV paper. If you do use it, please cite:

"Discriminative Learning of Deep Convolutional Feature Point Descriptors"  
Edgar Simo-Serra, Eduard Trulls, Luis Ferraz, Iasonas Kokkinos, Pascal Fua and Francesc Moreno-Noguer  
International Conference on Computer Vision (ICCV), 2015

The code is based on the [Torch7](http://torch.ch) framework.

## Overview

We learn compact discriminative feature point descriptors using a convolutional
neural network. We directly optimize for using L2 distance by training with a
pair of corresponding and non-corresponding patches correspond to small and
large distances respectively using a Siamese architecture. We deal with the
large number of potential pairs with the combination of a stochastic sampling
of the training set and an aggressive mining strategy biased towards patches
that are hard to classify. The resulting descriptor is 128 dimensions that can
be used as a drop-in replacement for any task involving SIFT. We show that this
descriptor generalizes well to various datasets.

See [the website](http://hi.cs.waseda.ac.jp/~esimo/research/deepdesc/) for more
detailed information information.

## License

```
Copyright (C) <2016> <Edgar Simo-Serra, Eduard Trulls>

This work is licensed under the Creative Commons
Attribution-NonCommercial-ShareAlike 4.0 International License. To view a copy
of this license, visit http://creativecommons.org/licenses/by-nc-sa/4.0/ or
send a letter to Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.

Edgar Simo-Serra, Waseda University, February 2016.
esimo@aoni.waseda.jp, http://hi.cs.waseda.ac.jp/~esimo/
Eduard Trulls, EPFL, February 2016.
eduard.trulls@epfl.ch, http://cvlabwww.epfl.ch/~trulls/
```

## Models

Four different models are made avaiable. Best iteration is chosen with a
validation subset. Model and training procedure is the same for all models,
only the training data varies. If not sure what model to use, use
`models/CNN3_p8_n8_split4_073000.t7`.

* `models/CNN3_p8_n8_split1_072000.t7`: Trained on Liberty and Yosemite.
* `models/CNN3_p8_n8_split2_104000.t7`: Trained on Liberty and Notre Dame.
* `models/CNN3_p8_n8_split3_067000.t7`: Trained on Yosemite and Notre Dame.
* `models/CNN3_p8_n8_split4_073000.t7`: Trained on a subset of Liberty, Yosemite, and Notre Dame.

## Usage

### Torch

See `example.lua` for the full example file.

Load a model:

```lua
model = torch.load( 'models/CNN3_p8_n8_split4_073000.t7' )
```

Normalize the patches, which should be a `Nx1x64x64` 4D float tensor with a range of 0-255:

```lua
for i=1,patches:size(1) do
   patches[i] = patches[i]:add( -model.mean ):cdiv( model.std )
end

```

Compute the 128-float descriptors for all the N patches:

```lua
descriptors = model.desc:forward( patches )
```

Note the output will be a `Nx128` 2D float tensor where each row is a descriptor.

### Matlab

It is possible to use Matlab by calling torch. This also requires the
`mattorch` package to work. Please look at the files in `matlab/`. In
particular, by calling `matlab/desc.lua` from Matlab, batches of descriptors
can be processed. This is done by using the code in `matlab/example.m`:

```matlab
patches = randn( 64, 64, 1, 2 );

save( 'patches.mat', 'patches' );
system( 'th desc.lua' );
desc = load( 'desc.mat' );

desc.x
```

As the Matlab matrix ordering is the opposite of Torch, please use the
`64x64x1xN` inputs with values in the 0-255 range.  Please note that this
creates temporary files `patches.mat` and `desc.mat` each time it is called.
You can also specify which model to use with:

```matlab
system( 'th desc.lua --model ../models/CNN3_p8_n8_split4_073000.t7' )
```

As this has a fair amount of overhead, use large batches to get best
performance.


## Citing

If you use this code please cite:

```
@InProceedings{SimoSerraICCV2015,
   author    = {Edgar Simo-Serra and Eduard Trulls and Luis Ferraz and Iasonas Kokkinos and Pascal Fua and Francesc Moreno-Noguer},
   title     = {{Discriminative Learning of Deep Convolutional Feature Point Descriptors}},
   booktitle = "Proceedings of the International Conference on Computer Vision (ICCV)",
   year      = 2015,
}
```

## Notes

Models are trained from scratch and not the models used in the paper as there
was an incompatibility with newer torch versions. Results should be comparable
in all cases.





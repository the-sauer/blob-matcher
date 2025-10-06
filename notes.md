# Notes

- Figure out how orientation is processed, (Is it just used to generate the patch?)
  - `modules/ptn/pytorch/models.py:248`: `ptnOrient` rotates the patch according to the orientation
- Max pooling over rotation
  - Max pooling has to be done over the first spatial dimension

## Augmentation and Perturbation

- `Augmentor` object randomly jitters locations, scales and orientations.
  - Configuration value `TRAINING.SOFT_AUG` turns this on and off.
- `Hardnet.forward` can also work with patches directly if the argument `theta == None`
- `args.hard_augm` (turned off by default) controls whether first dimension is flipped.

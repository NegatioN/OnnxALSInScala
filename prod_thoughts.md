## Prod Thoughts

1. each input node would (potentially) need its own map from a string representation
to the internal representation, as this cannot be done inside ONNX atm. (e.g. userId --> embedding index)
2. There would need to be a specific preprocessing and post-processing interface for each model-type if hosted from Scala (?)
Arbitrary input's seem difficult to use, and the pre/post processing seem impossible to package inside ONNX.
# Host an ALS model in Scala

### Install
1. For model creation: `micromamba create -f mamba_env.yml` (might have to remove cudatoolkit without gpu)
2. JVM + Scala to run on.

### Run
`micromamba activate onnx` and run model_export.ipynb


### TODO
1. ~~Export a real ALS model into pytorch~~
2. ~~Save model as ONNX~~
3. ~~Load an ONNX model in a Scala runtime~~
4. How would a prod format of a model look like?
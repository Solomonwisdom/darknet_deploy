#! /bin/bash

validDir=/workspace/darknet_deploy/darknet

$validDir/darknet detector valid /workspace/darknet_deploy/data/js_detect.data /workspace/darknet_deploy/data/js_detect_valid.cfg /workspace/darknet_deploy/data/js_detect_final.weights_best -out " " -thresh .5


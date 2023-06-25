# Automatic rock identification algolithm
By Yuta Shimizu†, Hideaki Miyamoto†, Patrick Michel†† <br>
† Department of Systems Innovation, The University of Tokyo <br>
†† Universite Côte d’Azur, Observatoire de la Côte d’Azur, Centre National de la Recherche Scientifique, Laboratoire Lagrange 

This repo is the official implementation of Shimizu et al., submitted. It is based on [CBNetV2](https://github.com/VDIGPKU/CBNetV2).

Contact us with hm@sys.t.u-tokyo.ac.jp

### Installation
Please refer to [CBNetV2](https://github.com/VDIGPKU/CBNetV2) and [MMDetection](https://github.com/open-mmlab/mmdetection) for installation and dataset preparation.
Our performing environment was:
- PyTorch ver. 1.7.0
- Cuda ver. 11.0
- cuDNN ver. 8

### Perform automatic rock identification
```
# Prepare dataset
python3 tools/dataset_converters/images2coco.py /path/to/test_dataset/ /path/to/list_dummy.txt annotations.json

# Perform identification
python3 ./tools/test.py configs/boulder/cascade_mask_rcnn_cbnet1.py /path/to/weight --show-dir /path/to/test_dataset/ --cfg-options data.test.ann_file="/path/to/annotations/annotations.json"
```

## License
The project is only free for academic research purposes, but needs authorization for commerce. For commerce permission, please contact hm@sys.t.u-tokyo.ac.jp.

# Model-Inversion-Attribute-Inference-Attack
 [USENIX'Sec 2022] Are your sensitive attributes private? novel model inversion attribute inference attacks on classification models
 
 Re-implementation of Model Inversion Attribute Inference (MIAI) attack with Pytorch

## How to use
```
python train_target_model.py
python run_attack.py
```

## Results
![这是图片](/image/my_result.png "result on Adult with DNN")

## Original repository implemented by BigML
[https://github.com/smehnaz/Black-box-Model-Inversion-Attribute-Inference](https://github.com/smehnaz/Black-box-Model-Inversion-Attribute-Inference)

## Reference
```
@inproceedings{mehnaz2022your,
  title={Are your sensitive attributes private? novel model inversion attribute inference attacks on classification models},
  author={Mehnaz, Shagufta and Dibbo, Sayanton V and De Viti, Roberta and Kabir, Ehsanul and Brandenburg, Bj{\"o}rn B and Mangard, Stefan and Li, Ninghui and Bertino, Elisa and Backes, Michael and De Cristofaro, Emiliano and others},
  booktitle={31st USENIX Security Symposium (USENIX Security 22)},
  pages={4579--4596},
  year={2022}
}
```
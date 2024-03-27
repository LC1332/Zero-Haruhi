## 使用说明

### 数据
https://huggingface.co/datasets/silk-road/ragged_CharacterEval \
https://huggingface.co/datasets/silk-road/CharacterEval2sharegpt


### 运行
1. git clone --recurse-submodules https://github.com/LC1332/Zero-Haruhi.git 或者 自行下载\
2. sh cp_file.sh\
3. cd LLaMA-Factory; sh llminfer_convert_charactereval.sh\
4. cd CharacterEval; sh eval_score.sh

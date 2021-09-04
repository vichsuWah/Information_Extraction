# *Cinnamon* - Document Information Extraction
> NTU ADL(2020 Spring) Final Project

## Intro.
The purpose of this task is to extract the important information from the offical documents.

Report: [Report Link](https://github.com/vichsuWah/Information_Extraction/blob/main/Report.pdf)

## Training
```cmd=
# train naive baseline
> make wu_train_naive_baseline_v2

# train blstm
> make wu_train_blstm_v2
```

## Evaluation
```cmd=
# inference naive baseline
> make wu_inference_naive_baseline_dev
> make wu_inference_naive_baseline_test

# inference blstm
> make wu_inference_blstm_dev_v2
> make wu_inference_blstm_test_v2
```

## Testing
```cmd= 
> python3 test.py {past_to_test_set_dir}
```


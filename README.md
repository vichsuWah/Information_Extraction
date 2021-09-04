# Cinnamon Information Extraction
ADL2020-SPRING Final project

## Report 
[[pdf]](https://github.com/wubinary/Information_Extraction/blob/master/Report.pdf)

## Train v2 preprocessing
```cmd=
# train naive baseline
> make wu_train_naive_baseline_v2

# train blstm
> make wu_train_blstm_v2
```

## Inference v2 preprocessing
```cmd=
# inference naive baseline
> make wu_inference_naive_baseline_dev
> make wu_inference_naive_baseline_test

# inference blstm
> make wu_inference_blstm_dev_v2
> make wu_inference_blstm_test_v2
```

## Test
```cmd= 
> python3.6 test.py {past_to_test_set_dir}

```


## Table A. Full training set

| Dataset | Paper | Ours k=2 optimistic | Top-2 contains | k=1 | k=2 random | k=2 nearest | TF-IDF+LR | Dup rate | k=1 clean | LR clean |
|---|---|---|---|---|---|---|---|---|---|---|
| DengueFilipino (n=346) | 0.998 | 0.939 | 0.939 | 0.864 | 0.824±0.010 | 0.864 | 0.988 | 0.026 | 0.861 | 0.988 |
| KirundiNews (n=923) | 0.905 | 0.927 | 0.927 | 0.892 | 0.832±0.006 | 0.892 | 0.901 | 0.859 | 0.800 | 0.908 |
| R8 (n=2189) | 0.954 | 0.952 | 0.952 | 0.915 | 0.903±0.001 | 0.915 | 0.967 | 0.002 | 0.915 | 0.967 |
| KinyarwandaNews (n=600) | 0.891 | 0.890 | 0.890 | 0.828 | 0.829±0.005 | 0.828 | 0.890 | 0.522 | 0.690 | 0.808 |
| SwahiliNews (n=300) | 0.927 | 0.943 | 0.943 | 0.857 | 0.858±0.007 | 0.857 | 0.897 | 0.013 | 0.855 | 0.895 |

## Table B. k sensitivity (random tie-break, mean of 5 seeds)

| Dataset | k=1 | k=2 | k=3 | k=5 | k=7 | k=3 opt | k=5 opt |
|---|---|---|---|---|---|---|---|
| DengueFilipino | 0.864 | 0.824 | 0.875 | 0.902 | 0.913 | 0.916 | 0.922 |
| KirundiNews | 0.892 | 0.832 | 0.811 | 0.824 | 0.809 | 0.819 | 0.863 |
| R8 | 0.915 | 0.903 | 0.921 | 0.927 | 0.922 | 0.937 | 0.936 |
| KinyarwandaNews | 0.828 | 0.829 | 0.834 | 0.823 | 0.822 | 0.867 | 0.838 |
| SwahiliNews | 0.857 | 0.858 | 0.905 | 0.895 | 0.903 | 0.913 | 0.900 |

## Table C. 5-shot (mean ± 95% CI over 5 trials)

| Dataset | Paper gzip | Ours k=2 optimistic | k=1 | k=2 random | k=2 nearest | TF-IDF+LR |
|---|---|---|---|---|---|---|
| KinyarwandaNews | 0.458 | 0.437±0.045 | 0.322±0.038 | 0.269±0.037 | 0.322±0.038 | 0.508±0.020 |
| KirundiNews | 0.541 | 0.475±0.028 | 0.359±0.038 | 0.309±0.028 | 0.359±0.038 | 0.540±0.040 |
| DengueFilipino | 0.652 | 0.649±0.044 | 0.521±0.049 | 0.409±0.044 | 0.521±0.049 | 0.746±0.050 |
| SwahiliNews | 0.627 | 0.626±0.022 | 0.460±0.025 | 0.410±0.022 | 0.460±0.025 | 0.621±0.062 |

## Table D. BGL_2k alert detection (mean over 5 splits)

| Split | majority acc | gzip k=1 F1 | gzip k=2 opt F1 | gzip k=2 rand F1 | TF-IDF+LR F1 |
|---|---|---|---|---|---|
| random | 0.927 | 0.969 | 0.993 | 0.979 | 0.979 |
| template_disjoint | 0.930 | 0.588 | 0.755 | 0.572 | 0.546 |

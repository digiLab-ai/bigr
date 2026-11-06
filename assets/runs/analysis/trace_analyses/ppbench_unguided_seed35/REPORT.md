# Trace analyses — ppbench_unguided_seed35

## rules

```
               mean  std
terminal_q   0.7381  NaN
path_sum     0.7381  NaN
plurality    0.5714  NaN
oracle       0.7857  NaN
base         0.5685  NaN
n_puzzles   42.0000  NaN
```

## e4b

```
         ece       auroc    
        mean std    mean std
step                        
3     0.0639 NaN  0.9862 NaN
6     0.0767 NaN  0.9943 NaN
9     0.0648 NaN  0.9941 NaN
12    0.0562 NaN  0.9963 NaN
15    0.0445 NaN  0.9963 NaN
18    0.0404 NaN  0.9963 NaN
21    0.0373 NaN  0.9972 NaN
24    0.0337 NaN  0.9963 NaN
27    0.0332 NaN  0.9925 NaN
30    0.0392 NaN  0.9927 NaN
33    0.0281 NaN  0.9935 NaN
36    0.0294 NaN  0.9934 NaN
39    0.0286 NaN  0.9920 NaN
42    0.0268 NaN  0.9909 NaN
45    0.0306 NaN  0.9900 NaN
48    0.0339 NaN  0.9895 NaN
```

## e10

```
      terminal_q     path_sum    
            mean std     mean std
scale                            
0.0       0.7381 NaN   0.7381 NaN
0.5       0.7470 NaN   0.7381 NaN
1.0       0.7470 NaN   0.7381 NaN
2.0       0.7440 NaN   0.7411 NaN
4.0       0.7232 NaN   0.7440 NaN
8.0       0.6667 NaN   0.7589 NaN
12.0      0.6280 NaN   0.7262 NaN
16.0      0.6429 NaN   0.7143 NaN
24.0      0.6071 NaN   0.6935 NaN
32.0      0.5833 NaN   0.6577 NaN
```

## ea7

```
     accuracy     terminal_acc_of_stopped     mean_stop_step     compute_frac    
         mean std                    mean std           mean std         mean std
tau                                                                              
0.90   0.6905 NaN                  0.7381 NaN        12.5238 NaN       0.2609 NaN
0.95   0.7143 NaN                  0.7381 NaN        13.5952 NaN       0.2832 NaN
0.99   0.7619 NaN                  0.7381 NaN        17.3571 NaN       0.3616 NaN
```


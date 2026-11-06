# Trace analyses — ppbench_unguided_seed32

## rules

```
               mean  std
terminal_q   0.7381  NaN
path_sum     0.7619  NaN
plurality    0.5952  NaN
oracle       0.8095  NaN
base         0.5714  NaN
n_puzzles   42.0000  NaN
```

## e4b

```
         ece       auroc    
        mean std    mean std
step                        
3     0.0543 NaN  0.9807 NaN
6     0.0850 NaN  0.9910 NaN
9     0.0842 NaN  0.9963 NaN
12    0.0670 NaN  0.9959 NaN
15    0.0553 NaN  0.9949 NaN
18    0.0539 NaN  0.9943 NaN
21    0.0504 NaN  0.9940 NaN
24    0.0425 NaN  0.9905 NaN
27    0.0361 NaN  0.9947 NaN
30    0.0380 NaN  0.9929 NaN
33    0.0411 NaN  0.9895 NaN
36    0.0438 NaN  0.9932 NaN
39    0.0383 NaN  0.9909 NaN
42    0.0396 NaN  0.9890 NaN
45    0.0333 NaN  0.9916 NaN
48    0.0314 NaN  0.9897 NaN
```

## e10

```
      terminal_q     path_sum    
            mean std     mean std
scale                            
0.0       0.7381 NaN   0.7619 NaN
0.5       0.7321 NaN   0.7619 NaN
1.0       0.7381 NaN   0.7619 NaN
2.0       0.7381 NaN   0.7530 NaN
4.0       0.7411 NaN   0.7381 NaN
8.0       0.6845 NaN   0.7292 NaN
12.0      0.6488 NaN   0.7054 NaN
16.0      0.6280 NaN   0.7173 NaN
24.0      0.6161 NaN   0.6905 NaN
32.0      0.5863 NaN   0.6786 NaN
```

## ea7

```
     accuracy     terminal_acc_of_stopped     mean_stop_step     compute_frac    
         mean std                    mean std           mean std         mean std
tau                                                                              
0.90   0.7143 NaN                  0.7381 NaN        13.3810 NaN       0.2788 NaN
0.95   0.7143 NaN                  0.7381 NaN        13.8571 NaN       0.2887 NaN
0.99   0.7381 NaN                  0.7381 NaN        17.2381 NaN       0.3591 NaN
```


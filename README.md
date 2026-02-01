# C-Cuda-learning


```bash
gcc helloworld.c -o hello_out
gcc train.c engine.c neural.c -o autograd
```


## notes
 - rlhf actually trains a second model and uses that to determine the reward for a main model

 - rlhf but with the reward and negative reward get too close. ie
 'i'm happy' vs 'i am happy' 
 is very similar but will get negative penalty
 so rlhf with dynamic polyicy



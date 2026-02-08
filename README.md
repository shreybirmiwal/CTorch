# corch


```bash
gcc helloworld.c -o hello_out
gcc engine.c neuron.c torch_utils.c train.c -o autograd -g

# with debugger
lldb ./autograd
```




## notes
 - rlhf actually trains a second model and uses that to determine the reward for a main model

 - rlhf but with the reward and negative reward get too close. ie
 'i'm happy' vs 'i am happy' 
 is very similar but will get negative penalty
 so rlhf with dynamic polyicy



SMF shortest makespan first

database scheduling talk:
 - going inear through schedules for db transations has waits for things
 - more efficient to specicial order such that dependencies n shi work out best
 - use greedy approx to find SMF shortest makespan first (iteratively add the transaction that increases makespan the least)
- in this case (question???) we are allowed to reorder the stuff and still have

classifer to predict 'hot data items' --> predict how long editing each item will take --> use greedy algo

how does 
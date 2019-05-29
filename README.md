# Reinforcement Learning: Flappy bird
## Execution
```
python dqn.py <mode> <image_size> <weights yes or no>
```
**For instance:**
```
python dqn.py train 75 True
```


## Rank-based prioritization of memory

**Memory map** is our new replay memory. It is a dictionary which contains a
key used as reference also in the heap and a value which is the original
tuple of the replay memory.

**Memory_content** is the reversed dictionary of memory map.

**Mem_id** and **delete_id** are two counters used like a window to increase and
decrease the memory map size while the iteration number is increasing.

**Sampling:**
random sample from 32 quantiles.

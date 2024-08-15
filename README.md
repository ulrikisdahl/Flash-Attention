# Flash attention (v1) implementation


- Paper: https://arxiv.org/abs/2205.14135 


<br>


### Benchmark:
<h4> I compare the Flash Attention implementation with a pure pytorch implementation of the attention algorithm </h4>
- Pure torch attention CUDA time total: 64.838ms
<br>
- Flash Attention CUDA time total: 9.864ms

<br>

### Validity of imlementation:
The CUDA implementation is compared with a correct, pure pytorch implementation and a official pytorch implementation of the attention mechanism in [main.py](test/main.py). The tests show that the Flash Attention implementation is correct.

Sample results:
<pre>
All close test: True
Average value in torch-attention: 0.0013573728501796722
Average value in scaled_dot_prod-attention: 0.0013573728501796722
Average value in flash-attention: 0.0013573728501796722
</pre>

Run tests: ```python3 -m test.main```
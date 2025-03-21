# 导数  
f(x)=x+b  
公式 ： `d f(x)/d x`  
定义 ：  
自变量极小变化，对于因变量的变化影响程度
`lim h->0 (f(x+h)-f(x))/h`  
_偏导数_ ： 求谁的偏导数就将除了‘谁’以外的变量全部视为常量  
_链式求导_ ： `h'(x)=f'(g(x))*g'(x)`  
常用激活函数：_tanh（）_  
`numpy.tanh()`  
`d tanhx / dx  = 1 - tanh^2*x`  
所以往上一层的梯度则可求  
# 拓扑排序  
针对有向无环图，将各个顶点排序  
用于遍历各个参数节点  
反向遍历推导梯度grad

# tensor  
通过Torch.tensor实例化张量  

# MLP  
多层感知器  本例中采用全连接

# Neuron  
micrograd中构建为wx+b的模型，  w、b都通过.engine去反向梯度优化，w、b都有各自的梯度

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from micrograd.engine import Value

a = Value(-4.0)
b = Value(2.0)
c = a + b
d = a * b + b**3
c += c + 1
c += 1 + c + (-a)
d += d * 2 + (b + a).relu()
d += 3 * d + (b - a).relu()
e = c - d
f = e**2
g = f / 2.0
g += 10.0 / f
print(f'{g.data:.4f}') # prints 24.7041, the outcome of this forward pass
g.backward()
print(f'{a.grad:.4f}') # prints 138.8338, i.e. the numerical value of dg/da
print(f'{b.grad:.4f}') # prints 645.5773, i.e. the numerical value of dg/db

# 定义 x 轴的值
x = np.linspace(-2 * np.pi, 2 * np.pi, 400)

# 计算 tan 和 tanh 的值
tan_values = np.tan(x)
tanh_values = np.tanh(x)

# 创建一个新的图形
plt.figure(figsize=(12, 6))

# 绘制 tan 函数
plt.subplot(1, 2, 1)
plt.plot(x, tan_values, label='tan(x)')
plt.ylim(-10, 10)  # 限制 y 轴的范围以更好地显示 tan 函数
plt.title('tan(x)')
plt.xlabel('x')
plt.ylabel('tan(x)')
plt.legend()
plt.grid(True)

# 绘制 tanh 函数
plt.subplot(1, 2, 2)
plt.plot(x, tanh_values, label='tanh(x)', color='orange')
plt.title('tanh(x)')
plt.xlabel('x')
plt.ylabel('tanh(x)')
plt.legend()
plt.grid(True)

# 显示图形
plt.tight_layout()
plt.show()
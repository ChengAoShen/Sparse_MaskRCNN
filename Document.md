# Torch中的稀疏矩阵

## 使用稀疏矩阵的原因

在一般情况下，PyTorch会将所有的元素连续储存在物理内存中，可以快速访问到需要的元素。各种稀疏存储格式如COO、CSR/CSC、半结构化、LIL等已经发展了多年。它们在确切的布局上有所不同，但它们都通过有效地表示零值元素来压缩数据。通过压缩重复零，稀疏存储格式旨在节省各种CPU和GPU的内存和计算资源。

---

## Sparse Semi-Structured Tensors

该特性在NVIDIA's Ampere上进行引入，又被称为细粒度稀疏结构或者2:4稀疏结构。

> 目前还是原型功能，不进行使用

---

## COO格式

分别储存非0元素的索引以及其值

* 索引放在大小为`(ndim,nse)`，元素类型为`torch.int64`的`incdices`张量中
* 相应的值收集在大小为 `(nse,)` 的 `values` 张量中，并且具有任意整数或浮点数元素类型

### 构建

```python
i = [[0, 1, 1],
     [2, 0, 2]]
v =  [3, 4, 5]
s = torch.sparse_coo_tensor(i, v, (2, 3))
s.to_dense()

# 普通tensor转换为稀疏
t.to_sparse_coo()
```

### 稀疏和稠密混合使用

可以在索引部分使用稀疏的表示，然后之后的值本身也可以是一个张量

```python
i = [[0, 1, 1],
      [2, 0, 2]]
v = [[3, 4], [5, 6], [7, 8]]
s = torch.sparse_coo_tensor(i, v, (2, 3, 2))
```

需要注意的是，如果存在两个相同的索引，则在合并之后会将其求和

```python
In [9]: s
Out[9]:tensor(indices=tensor([[0, 1, 1],
                              [2, 0, 0]]),
              values=tensor([3, 4, 5]),
              size=(2, 3), nnz=3, layout=torch.sparse_coo)

In [10]: s.to_dense()
Out[10]:tensor([[0, 0, 3],
                [9, 0, 0]])
```

### 使用方法

获得稀疏和稠密维度，注意需要在合并后的tensor使用

```python
s.sparse_dim(), s.dense_dim()
```

索引，按照每一级的索引->值的顺序进行

```python
In [17]: s
Out[17]:
tensor(indices=tensor([[0, 1, 1],
                       [2, 0, 0]]),
       values=tensor([3, 4, 5]),
       size=(2, 3), nnz=3, layout=torch.sparse_coo)

In [18]: s[1]
Out[18]:
tensor(indices=tensor([[0, 0]]),
       values=tensor([4, 5]),
       size=(3,), nnz=2, layout=torch.sparse_coo)
```

反向传播

```python
In [42]: i=[[0,1,1],[2,0,2]]
    ...: v=[3.0,4.0,5.0]
    ...: s=torch.sparse_coo_tensor(i,v,(2,3))
    ...: s.requires_grad=True
    ...: ans=s.sum()
    ...: ans.backward()
    ...: s.grad
Out[42]:
tensor([[1., 1., 1.],
        [1., 1., 1.]])
```

> 该类型主要问题：不能使用log运算/交叉熵运算/切片方法

---

## CSR格式

稀疏维度最多为2，使用 (B + M + K) 维张量来表示 N 维稀疏压缩混合张量，其中 B、M 和 K 分别是批量、稀疏和密集维度的数量。在一般情况下，(B + 2 + K) 维稀疏 CSR 张量由两个 (B + 1) 维索引张量 `crow_indices` 和 `col_indices` 以及 (1 + K) 维 `values` 张量组成。

* `crow_indices.shape == (*batchsize, nrows + 1)`：张量中的每个连续数字减去前面的数字表示给定行中的元素数量，最后一个元素是指定元素的数量
* `col_indices.shape == (*batchsize, nse)`：张量包含每个元素的列索引
* `values.shape == (nse, *densesize)`：包含 CSR 张量元素的值

```python
crow_indices = torch.tensor([0, 2, 4])
col_indices = torch.tensor([0, 1, 0, 1])
values = torch.tensor([1, 2, 3, 4])
csr = torch.sparse_csr_tensor(crow_indices, col_indices, values, dtype=torch.float64)
```

稀疏矩阵向量乘法可以使用 `tensor.matmul()` 方法来执行。这是目前 CSR 张量支持的唯一数学运算。

CSC张量：当转置涉及交换稀疏维度时，稀疏 CSC 张量本质上是稀疏 CSR 张量的转置。

* `ccol_indices` 张量由压缩的列索引组成。这是形状为 `(*batchsize, ncols + 1)` 的 (B + 1)-D 张量。最后一个元素是指定元素的数量 `nse` 。该张量根据给定列的开始位置对 `values` 和 `row_indices` 中的索引进行编码。张量中的每个连续数字减去前面的数字表示给定列中的元素数量。
* `row_indices` 张量包含每个元素的行索引。这是形状为 `(*batchsize, nse)` 的 (B + 1)-D 张量。
* `values` 张量包含 CSC 张量元素的值。这是形状为 `(nse, *densesize)` 的 (1 + K)-D 张量。

---

## BSR格式

稀疏 BSR（块压缩稀疏行）张量格式实现了用于存储二维张量的 BSR 格式，并扩展为支持批量稀疏 BSR 张量和作为多维张量块的值。

稀疏 BSR 张量由三个张量组成： `crow_indices` 、 `col_indices` 和 `values` ：

* `crow_indices` 张量由压缩的行索引组成。这是形状为 `(*batchsize, nrowblocks + 1)` 的 (B + 1)-D 张量。最后一个元素是指定块的数量 `nse` 。该张量根据给定列块的起始位置对 `values` 和 `col_indices` 中的索引进行编码。张量中的每个连续数字减去前面的数字表示给定行中的块数。
* `col_indices` 张量包含每个元素的列块索引。这是形状为 `(*batchsize, nse)` 的 (B + 1)-D 张量。
* `values` 张量包含收集到二维块中的稀疏 BSR 张量元素的值。这是形状为 `(nse, nrowblocks, ncolblocks, *densesize)` 的 (1 + 2 + K)-D 张量。

## BSC格式

稀疏 BSC（块压缩稀疏列）张量格式实现了用于存储二维张量的 BSC 格式，并扩展为支持批量稀疏 BSC 张量和作为多维张量块的值。

稀疏 BSC 张量由三个张量组成： `ccol_indices` 、 `row_indices` 和 `values` ：

* `ccol_indices` 张量由压缩的列索引组成。这是形状为 `(*batchsize, ncolblocks + 1)` 的 (B + 1)-D 张量。最后一个元素是指定块的数量 `nse` 。该张量根据给定行块的起始位置对 `values` 和 `row_indices` 中的索引进行编码。张量中的每个连续数字减去前面的数字表示给定列中的块数。
* `row_indices` 张量包含每个元素的行块索引。这是形状为 `(*batchsize, nse)` 的 (B + 1)-D 张量。
* `values` 张量包含收集到二维块中的稀疏 BSC 张量元素的值。这是形状为 `(nse, nrowblocks, ncolblocks, *densesize)` 的 (1 + 2 + K)-D 张量。

---

## 项目进展

- [x] 以简单形式跑通MaskRCNN
- [x] 测试不同的稀疏格式
- [x] 使用矩阵乘法代替切片索引（2D）
- [ ] 重写TorchVision`roi_heads.py`文件
- [x] dataloader实现多边形 **点->稀疏矩阵** 的快速转换





记录：找到了最后需要的裁剪索引box来自于proposal，proposal来自于RPN的输出，找到RPN输出的proposal格式

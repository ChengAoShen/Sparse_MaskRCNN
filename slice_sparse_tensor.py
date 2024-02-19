import torch


def slice_sparse_tensor(
    m: torch.Tensor, row_i: int, row_j: int, col_i: int, col_j: int
) -> torch.Tensor:
    """slice the 2d sparse tensor

    Args:
        m: sparse tensor
        row_i: begin row index
        row_j: end row index
        col_i: begin col index
        col_j: end col index

    Returns:
        sliced sparse tensor
    """
    indices = torch.tensor([[x, x + i] for x in range(row_j - row_i)]).t()
    values = torch.ones(row_j - row_i)
    row_m = torch.sparse_coo_tensor(indices, values, (row_j - row_i, m.shape[0]))

    indices = torch.tensor([[x + k, x] for x in range(col_j - col_i)]).t()
    values = torch.ones(col_j - col_i)
    col_m = torch.sparse_coo_tensor(indices, values, (n, col_j - col_i))

    return torch.sparse.mm(torch.sparse.mm(row_m, m), col_m)


if __name__ == "__main__":
    m, n = 10, 15
    indices = torch.tensor([[0, 1, 2], [2, 0, 4]])
    values = torch.tensor([3, 4, 5], dtype=torch.float32)
    M = torch.sparse_coo_tensor(indices, values, (m, n))
    print(M.to_dense())

    i, j = 0, 5
    k, p = 0, 3
    submatrix = slice_sparse_tensor(M, i, j, k, p)
    print(submatrix.to_dense())

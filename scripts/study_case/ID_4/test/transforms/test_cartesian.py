import torch
from scripts.study_case.pytorch_geometric_fork.torch_geometric.transforms import Cartesian
from scripts.study_case.pytorch_geometric_fork.torch_geometric.data import Data


def test_cartesian():
    assert Cartesian().__repr__() == 'Cartesian(norm=True, max_value=None)'

    pos = torch.Tensor([[-1, 0], [0, 0], [2, 0]])
    edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]])
    edge_attr = torch.Tensor([1, 1, 1, 1])

    data = Data(edge_index=edge_index, pos=pos)
    data = Cartesian(norm=False)(data)
    assert len(data) == 3
    assert data.pos.tolist() == pos.tolist()
    assert data.edge_index.tolist() == edge_index.tolist()
    assert data.edge_attr.tolist() == [[1, 0], [-1, 0], [2, 0], [-2, 0]]

    data = Data(edge_index=edge_index, pos=pos, edge_attr=edge_attr)
    data = Cartesian(norm=True)(data)
    assert len(data) == 3
    assert data.pos.tolist() == pos.tolist()
    assert data.edge_index.tolist() == edge_index.tolist()
    assert data.edge_attr.tolist() == [[1, 0.75, 0.5], [1, 0.25, 0.5],
                                       [1, 1, 0.5], [1, 0, 0.5]]

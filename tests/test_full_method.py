import torch
from v_prism import full_VPRISM_method



def test_full_method_throws_no_errors():
    num_classes = 3
    points = torch.tensor([
        [1.1, 0.2, 0.1],
        [1.1, 0.2, 0.2],
        [1.0, 0.1, 0.1],
        [1.0, 0.1, 0.0],
        [1.0, 0.3, 0.3],
        [1.0, 0.2, 0.5],
        [1.0, 0.4, 0.1],
        [1.0, 0.4, 0.2],
        [1.0, 0.5, 0.2],
    ])
    P = points.shape[0]
    mask = torch.tensor([0, 0, 0, 0, 1, 1, 2, 2, 2]).to(torch.int64)
    camera_pos = torch.zeros(3)

    vprism_map = full_VPRISM_method(points, mask, num_classes, camera_pos)

    preds = vprism_map.predict(points)

    assert preds.shape == (P, num_classes)




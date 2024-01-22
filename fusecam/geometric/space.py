import torch

class SpatialVolumeMetric(torch.nn.Module):
    def __init__(self, origin, step_size, orientation, skewed_axis_matrix=None):
        super(SpatialVolumeMetric, self).__init__()
        # Convert inputs to tensors, using clone().detach() for existing tensors
        self.origin = self.to_tensor(origin)
        self.step_size = self.to_tensor(step_size)
        self.orientation = self.to_tensor(orientation)
        if skewed_axis_matrix is None:
            skewed_axis_matrix = torch.eye(3)
        self.skewed_axis_matrix = self.to_tensor(skewed_axis_matrix)

    def to_tensor(self, data, size=3):
        if not isinstance(data, torch.Tensor):
            data = torch.tensor(data, dtype=torch.float32)
        if data.size(-1) != size:
            raise ValueError(f"Expected size {size}, but got {data.size(0)}")
        return data.clone().detach()

    def to_lab_coordinates(self, tensor_indices):
        tensor_indices = self.to_tensor(tensor_indices)

        # Apply skewed axis transformation
        skewed_indices = torch.matmul(self.skewed_axis_matrix, tensor_indices.unsqueeze(-1)).squeeze(-1)

        # Apply step size to each index component
        scaled_indices = self.step_size * skewed_indices

        # Apply orientation transformation
        return self.origin + torch.matmul(self.orientation, scaled_indices.unsqueeze(-1)).squeeze(-1)

    def to_tensor_indices(self, lab_coordinates):
        lab_coordinates = self.to_tensor(lab_coordinates)

        # Subtract the origin and apply the inverse orientation transformation
        transformed_coords = lab_coordinates - self.origin
        transformed_coords = torch.matmul(torch.inverse(self.orientation), transformed_coords.unsqueeze(-1)).squeeze(-1)

        # Divide by the step size
        tensor_indices = transformed_coords / self.step_size

        # Apply the inverse skewed axis transformation
        return torch.matmul(torch.inverse(self.skewed_axis_matrix), tensor_indices.unsqueeze(-1)).squeeze(-1)


class SpatialPlaneMetric(torch.nn.Module):
    def __init__(self, origin, step_size, orientation, skewed_axis_matrix=None):
        super(SpatialPlaneMetric, self).__init__()
        # Convert inputs to tensors, using clone().detach() for existing tensors
        self.origin = self.to_tensor(origin, size=2)
        self.step_size = self.to_tensor(step_size, size=2)
        self.orientation = self.to_tensor(orientation, size=2)

        if skewed_axis_matrix is None:
            skewed_axis_matrix = torch.eye(2)
        self.skewed_axis_matrix = self.to_tensor(skewed_axis_matrix, size=2)

    def to_tensor(self, data, size=2):
        if not isinstance(data, torch.Tensor):
            data = torch.tensor(data, dtype=torch.float32)
        if data.size(-1) != size:
            raise ValueError(f"Expected size {size}, but got {data.size(0)}")
        return data.clone().detach()

    def to_plane_coordinates(self, tensor_indices):
        tensor_indices = self.to_tensor(tensor_indices, size=2)

        # Apply skewed axis transformation
        skewed_indices = torch.matmul(self.skewed_axis_matrix, tensor_indices.unsqueeze(-1)).squeeze(-1)

        # Apply step size to each index component
        scaled_indices = self.step_size * skewed_indices

        # Apply orientation transformation
        return self.origin + torch.matmul(self.orientation, scaled_indices.unsqueeze(-1)).squeeze(-1)

    def to_tensor_indices(self, plane_coordinates):
        plane_coordinates = self.to_tensor(plane_coordinates, size=2)

        # Subtract the origin and apply the inverse orientation transformation
        transformed_coords = plane_coordinates - self.origin
        transformed_coords = torch.matmul(torch.inverse(self.orientation), transformed_coords.unsqueeze(-1)).squeeze(-1)

        # Divide by the step size
        tensor_indices = transformed_coords / self.step_size

        # Apply the inverse skewed axis transformation
        return torch.matmul(torch.inverse(self.skewed_axis_matrix), tensor_indices.unsqueeze(-1)).squeeze(-1)

import torch

class SpatialVolumeMetric(torch.nn.Module):
    """
    A class to represent and manipulate spatial metrics for a 3D volume, including rotation, translation, and skew transformations.

    Attributes:
        origin (torch.Tensor): The origin point in lab coordinates for rotation.
        step_size (torch.Tensor): The scaling factor for each axis, accounting for voxel size.
        orientation (torch.Tensor): The orientation matrix for aligning tensor indices with lab coordinates.
        translation (torch.Tensor): Additional translation applied after rotation and before scaling.
        skewed_axis_matrix (torch.Tensor): The matrix to transform tensor indices to account for skewed axes.
    """

    def __init__(self, origin, step_size, orientation, translation, skewed_axis_matrix=None):
        """
        Initializes the SpatialVolumeMetric with given parameters.

        Parameters:
            origin (array-like): The origin point in (fractional) tensor indices for rotation.
            step_size (array-like): The scaling factor for each axis.
            orientation (array-like): The orientation matrix for the volume.
            translation (array-like): Additional translation applied in tensor indices.
            skewed_axis_matrix (array-like, optional): The matrix for skewed axes transformation. Defaults to an identity matrix if None.
        """
        super(SpatialVolumeMetric, self).__init__()
        self.origin = self.to_tensor(origin)
        self.step_size = self.to_tensor(step_size)
        self.orientation = self.to_tensor(orientation)
        self.translation = self.to_tensor(translation)  # New translation attribute
        if skewed_axis_matrix is None:
            skewed_axis_matrix = torch.eye(3)
        self.skewed_axis_matrix = self.to_tensor(skewed_axis_matrix)

    def to_tensor(self, data, size=3):
        """
        Converts the input data to a tensor and ensures it has the correct size.

        Parameters:
            data (array-like or torch.Tensor): The data to be converted to a tensor.
            size (int): The expected size of the last dimension of the tensor.

        Returns:
            torch.Tensor: The converted tensor with the specified size.
        """
        if not isinstance(data, torch.Tensor):
            data = torch.tensor(data, dtype=torch.float32)
        if data.size(-1) != size:
            raise ValueError(f"Expected size {size}, but got {data.size(0)}")
        return data.clone().detach()

    def to_lab_coordinates(self, tensor_indices):
        """
        Converts tensor indices to lab coordinates considering the skewed axes, step size, orientation, and translation.

        Parameters:
            tensor_indices (array-like): The indices in the tensor to be converted.

        Returns:
            torch.Tensor: The corresponding lab coordinates.
        """
        tensor_indices = self.to_tensor(tensor_indices)

        # Translate tensor indices so that the origin is at (0,0,0)
        translated_indices = tensor_indices - self.origin

        # Apply orientation (rotation) transformation
        rotated_indices = torch.matmul(self.orientation, translated_indices.unsqueeze(-1)).squeeze(-1)

        # Translate back and apply additional translation
        back_translated_indices = rotated_indices + self.origin
        final_translated_indices = back_translated_indices + self.translation

        # Apply skewed axis transformation
        skewed_indices = torch.matmul(self.skewed_axis_matrix, final_translated_indices.unsqueeze(-1)).squeeze(-1)

        # Apply step size to each index component
        scaled_indices = self.step_size * skewed_indices

        return scaled_indices

    def to_tensor_indices(self, lab_coordinates):
        """
        Converts lab coordinates back to tensor indices considering the inverse transformations, including translation.

        Parameters:
            lab_coordinates (array-like): The coordinates in the lab to be converted.

        Returns:
            torch.Tensor: The corresponding tensor indices.
        """
        lab_coordinates = self.to_tensor(lab_coordinates)

        # Inverse step size and skewed axis transformation
        tensor_indices = lab_coordinates / self.step_size
        unskewed_indices = torch.matmul(torch.inverse(self.skewed_axis_matrix), tensor_indices.unsqueeze(-1)).squeeze(-1)

        # Subtract additional translation and translate to rotate around the origin
        final_translated_indices = unskewed_indices - self.translation
        translated_indices = final_translated_indices - self.origin

        # Apply inverse orientation (rotation) transformation
        rotated_indices = torch.matmul(torch.inverse(self.orientation), translated_indices.unsqueeze(-1)).squeeze(-1)

        # Translate back to the original coordinate system
        return rotated_indices + self.origin


class SpatialPlaneMetric(torch.nn.Module):
    """
    A class to represent and manipulate spatial metrics for a 2D plane, including rotation, translation, and skew transformations.

    Attributes:
        origin (torch.Tensor): The origin point in plane coordinates for rotation.
        step_size (torch.Tensor): The scaling factor for each axis, accounting for pixel size.
        orientation (torch.Tensor): The orientation matrix for aligning tensor indices with plane coordinates.
        translation (torch.Tensor): Additional translation applied after rotation and before scaling.
        skewed_axis_matrix (torch.Tensor): The matrix to transform tensor indices to account for skewed axes.
    """

    def __init__(self, origin, step_size, orientation, translation, skewed_axis_matrix=None):
        """
        Initializes the SpatialPlaneMetric with given parameters.

        Parameters:
            origin (array-like): The origin point in plane coordinates for rotation.
            step_size (array-like): The scaling factor for each axis.
            orientation (array-like): The orientation matrix for the plane.
            translation (array-like): Additional translation applied in tensor indices.
            skewed_axis_matrix (array-like, optional): The matrix for skewed axes transformation. Defaults to an identity matrix if None.
        """
        super(SpatialPlaneMetric, self).__init__()
        self.origin = self.to_tensor(origin, size=2)
        self.step_size = self.to_tensor(step_size, size=2)
        self.orientation = self.to_tensor(orientation, size=2)
        self.translation = self.to_tensor(translation, size=2)
        if skewed_axis_matrix is None:
            skewed_axis_matrix = torch.eye(2)
        self.skewed_axis_matrix = self.to_tensor(skewed_axis_matrix, size=2)

    def to_tensor(self, data, size=2):
        """
        Converts the input data to a tensor and ensures it has the correct size for 2D operations.

        Parameters:
            data (array-like or torch.Tensor): The data to be converted to a tensor.
            size (int): The expected size of the last dimension of the tensor.

        Returns:
            torch.Tensor: The converted tensor with the specified size.
        """
        if not isinstance(data, torch.Tensor):
            data = torch.tensor(data, dtype=torch.float32)
        if data.size(-1) != size:
            raise ValueError(f"Expected size {size}, but got {data.size(0)}")
        return data.clone().detach()

    def to_plane_coordinates(self, tensor_indices):
        """
        Converts tensor indices to plane coordinates considering the skewed axes, step size, orientation, and translation.

        Parameters:
            tensor_indices (array-like): The indices in the tensor to be converted.

        Returns:
            torch.Tensor: The corresponding plane coordinates.
        """
        tensor_indices = self.to_tensor(tensor_indices, size=2)

        # Translate tensor indices so that the origin is at (0,0)
        translated_indices = tensor_indices - self.origin

        # Apply orientation (rotation) transformation
        rotated_indices = torch.matmul(self.orientation, translated_indices.unsqueeze(-1)).squeeze(-1)

        # Translate back and apply additional translation
        back_translated_indices = rotated_indices + self.origin
        final_translated_indices = back_translated_indices + self.translation

        # Apply skewed axis transformation
        skewed_indices = torch.matmul(self.skewed_axis_matrix, final_translated_indices.unsqueeze(-1)).squeeze(-1)

        # Apply step size to each index component
        scaled_indices = self.step_size * skewed_indices

        return scaled_indices

    def to_tensor_indices(self, plane_coordinates):
        """
        Converts plane coordinates back to tensor indices considering the inverse transformations, including translation.

        Parameters:
            plane_coordinates (array-like): The coordinates in the plane to be converted.

        Returns:
            torch.Tensor: The corresponding tensor indices.
        """
        plane_coordinates = self.to_tensor(plane_coordinates, size=2)

        # Inverse step size and skewed axis transformation
        tensor_indices = plane_coordinates / self.step_size
        unskewed_indices = torch.matmul(torch.inverse(self.skewed_axis_matrix), tensor_indices.unsqueeze(-1)).squeeze(-1)

        # Subtract additional translation and translate to rotate around the origin
        final_translated_indices = unskewed_indices - self.translation
        translated_indices = final_translated_indices - self.origin

        # Apply inverse orientation (rotation) transformation
        rotated_indices = torch.matmul(torch.inverse(self.orientation), translated_indices.unsqueeze(-1)).squeeze(-1)

        # Translate back to the original coordinate system
        return rotated_indices + self.origin

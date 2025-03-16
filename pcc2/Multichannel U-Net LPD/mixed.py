from typing import Literal

import numpy as np
import torch
from parallelproj import RegularPolygonPETProjector
from torch.utils.data import Dataset
from torchvision.transforms import functional as F

from utils.ellipsoids import generate_random_3d_ellipsoids
from utils.synthmorph import draw_perlin, labels_to_image, minmax_norm


class MixedDataset(Dataset):
    """A PyTorch Dataset class that generates 3D random ellipsoids and/or random shapes based phantoms."""

    def __init__(
        self,
        device: torch.device,
        projector: RegularPolygonPETProjector,
        n_imgs: int,
        n_labels: int = 6,
        noise_interval: tuple = (0.0, 1.0),
        kernel_size: int = 5,
        sigma: float = 2.0,
        modality: Literal["ellipsoids", "shapes", "mixed"] = "mixed",
    ):
        """Initialise the MixedShapes dataset.

        Args:
            device (torch.device): The device to use.
            proj (RegularPolygonPETProjector): The projector to use for the dataset.
            n_imgs (int): The number of objects to generate.
            n_labels (int, optional): The number of possible different labels in the label map. Defaults to 5.
            noise_interval (tuple, optional): The interval for the Poisson noise. Defaults to (0.0, 1.0) (no noise).
            kernel_size (int, optional): The size of the Gaussian blur kernel. Defaults to 5.
            sigma (float, optional): The standard deviation of the Gaussian blur kernel. Defaults to 2.0.
            modality (Literal["ellipsoids", "shapes", "mixed"], optional): The modality of the dataset. Defaults to "mixed".
                If "ellipsoids", only ellipsoids are generated. If "shapes", only shapes are generated. If "mixed", both are
                generated with equal probability.
        """
        self.device = device
        self.projector = projector

        # Image parameters
        self.n_imgs = n_imgs
        self.img_shape = projector.in_shape

        # Shapes parameters
        self.n_labels = n_labels

        # Noise parameters
        self.noise_interval = noise_interval

        # Guassian blur
        self.kernel_size = kernel_size
        self.sigma = sigma

        # Modality
        self.modality = modality

        # CT parameters - attenuation coefficients (cm^-1)
        self.ct_coefficients = {
            "air": 0.0,
            "soft_tissue": 0.22,  # General soft tissue
            "fat": 0.18,          # Fat tissue
            "bone": 0.56,         # Bone (varies with density)
            "blood": 0.24,        # Blood
            "lung": 0.05,         # Lung tissue
            "muscle": 0.23,       # Muscle tissue
            "brain": 0.21         # Brain tissue
        }

        # List of tissue types for mapping
        self.tissue_types = ["soft_tissue", "bone", "fat", "lung", "blood", "muscle", "brain"]

        if self.modality not in ["ellipsoids", "shapes", "mixed"]:
            raise ValueError(
                "Invalid modality. Choose between 'ellipsoids', 'shapes' and 'mixed'."
            )

    def __getitem__(self, _) -> tuple:
        """Generates a single 3D random ellipsoid phantom.

        Args:
            _ (int): The index of the image to generate.

        Returns:
            Tuple: The generated sinogram, the image tensors, the label map tensor and the labels mapping dict.
        """
        sino, img, ct_img,labels, mapping = self.generate_shapes()

        return sino, img, ct_img, labels, mapping

    def __len__(self) -> int:
        """Returns the number of images in the dataset.

        Returns:
            int: Number of images in the dataset.
        """
        return self.n_imgs

    def _create_image_from_label_maps(self, label_map: torch.Tensor) -> tuple:
        """Creates an image from the label map.

        Args:
            label_map (torch.Tensor): The label map to convert.

        Returns:
            dict: The generated image, CT image and the label map.
        """
        gen = labels_to_image(self.device, label_map)
        gen_img = gen["image"]
        gen_label_map = gen["labels"]
        gen_mapping = gen["mapping"]

        # Generate CT image from the label map
        gen_ct_img = self._generate_ct_image(gen_label_map)

        # Combine the two label maps
        gen_label_map = self._pad_label_maps(gen_label_map, self.n_labels * 2)
        gen_mapping = torch.nn.functional.pad(
            gen_mapping,
            (0, self.n_labels * 2 - gen_mapping.shape[0]),
            mode="constant",
            value=-42,
        )

        return gen_img, gen_ct_img, gen_label_map, gen_mapping
    

    def _generate_ct_image(self, label_map: torch.Tensor) -> torch.Tensor:
        """Generates a CT image based on the label map.
        
        Maps labels to different tissue types and assigns appropriate attenuation coefficients.
        The mapping is deterministic but ensures consistent tissue assignment for each label.

        Args:
            label_map (torch.Tensor): The label map to convert.

        Returns:
            torch.Tensor: The generated CT image with attenuation coefficients.
        """
            # Check the dimension of label_map
        if len(label_map.shape) == 4:
            # 4D case: (n_labels, x, y, z)
            n_labels, x, y, z = label_map.shape
            ct_image = torch.full((x, y, z), self.ct_coefficients["air"], 
                                dtype=torch.float32, device=self.device)
            
            # Process each label separately
            for l in range(n_labels):
                label_slice = label_map[l]
                if torch.any(label_slice > 0):  # Only process non-empty slices
                    # Determine tissue type for this label
                    tissue_idx = ((l+1) * 1234) % len(self.tissue_types)
                    tissue_type = self.tissue_types[tissue_idx]
                    
                    # Get base attenuation coefficient for this tissue
                    base_mu = self.ct_coefficients[tissue_type]
                    
                    # Add small variation to make it more realistic (±10%)
                    variation = (((l+1) * 5678) % 21 - 10) / 100  # -10% to +10%
                    mu_value = base_mu * (1 + variation)
                    
                    # Apply the attenuation coefficient where label is present
                    mask = (label_slice > 0)
                    ct_image[mask] = mu_value
                        
        else:
            # Initialize CT image with air attenuation coefficient
            ct_image = torch.full(self.img_shape, self.ct_coefficients["air"], 
                                dtype=torch.float32, device=self.device)

            # Get unique labels (excluding background which is 0)
            unique_labels = torch.unique(label_map)
            
            # For each label in the label map, assign an appropriate attenuation coefficient
            for label_idx in unique_labels:
                if label_idx == 0:  # Skip background
                    continue
                    
                # Assign tissue type based on label index (deterministic)
                tissue_idx = (int(label_idx) * 1234) % len(self.tissue_types)
                tissue_type = self.tissue_types[tissue_idx]
                
                # Get base attenuation coefficient for this tissue
                base_mu = self.ct_coefficients[tissue_type]
                
                # Add small variation to make it more realistic (±10%)
                variation = ((int(label_idx) * 5678) % 21 - 10) / 100  # -10% to +10%
                mu_value = base_mu * (1 + variation)
                    
                # Create mask for this label and apply the attenuation coefficient
                mask = (label_map == label_idx)
                ct_image[mask] = mu_value
        
        # Add small random noise to make the image more realistic
        noise = torch.randn_like(ct_image) * 0.005  # Reduced noise level for CT
        ct_image = ct_image + noise
        
        # Ensure non-negative values
        ct_image = torch.clamp_min(ct_image, 0.0)
        
        return ct_image

    def _generate_label_maps(self) -> torch.Tensor:
        """Generate the labels for the image.

        Returns:
            list: A list of label maps.
        """
        label_map = torch.zeros(*self.img_shape).to(self.device)

        while len(torch.unique(label_map)) < 2:  # Avoid empty label maps

            # Select the modality
            if self.modality == "ellipsoids":
                img_type = 0
            elif self.modality == "shapes":
                img_type = 1
            else:
                img_type = torch.rand(1)  # Randomly select the modality

            if img_type > 0.5:
                for _ in range(2):
                    # Generate the shapes using Perlin noise
                    im = draw_perlin(
                        device=self.device,
                        out_shape=(*self.img_shape, self.n_labels),
                        scales=[32, 64, 128],
                        max_std=1,
                    )
                    im = self._create_geodes(im)
                    label_map += torch.argmax(im, dim=0)

            else:
                for _ in range(2):
                    # Generate the ellipsoids
                    label_map += generate_random_3d_ellipsoids(
                        device=self.device,
                        diag_len=200,
                        out_shape=self.img_shape,
                        n_ellipsoids=self.n_labels - 1,
                    )
                # Substract ellipsoids to make holes
                label_map -= generate_random_3d_ellipsoids(
                    device=self.device,
                    diag_len=200,
                    out_shape=self.img_shape,
                    n_ellipsoids=np.random.randint(self.n_labels // 2, self.n_labels),
                )

            torch.clamp_(label_map, 0)

        return label_map

    def _create_geodes(self, perlin_noise: torch.Tensor) -> torch.Tensor:
        """Create a geodesic shape from the Perlin noise.

        Args:
            perlin_noise (torch.Tensor): The Perlin noise tensor.

        Returns:
            torch.Tensor: The geodesic shape tensor.
        """
        # Get dimensions
        n_labels, sizeX, sizeY, sizeZ = perlin_noise.shape

        # Center coordinates
        centerX = sizeX // 2
        centerY = sizeY // 2
        centerZ = sizeZ // 2

        # Create a grid of coordinates
        x_coords = torch.arange(sizeX).view(-1, 1, 1).repeat(1, sizeY, sizeZ)
        y_coords = torch.arange(sizeY).view(1, -1, 1).repeat(sizeX, 1, sizeZ)
        z_coords = torch.arange(sizeZ).view(1, 1, -1).repeat(sizeX, sizeY, 1)

        # Calculate squared distances
        distanceX = (centerX - x_coords) ** 2
        distanceY = (centerY - y_coords) ** 2
        distanceZ = (centerZ - z_coords) ** 2

        # Calculate Euclidean distance to the center
        distanceToCenter = torch.sqrt(
            distanceX + distanceY + distanceZ
        ) * np.random.uniform(1.5, 2.5)

        # Normalize the distance
        distanceToCenter = distanceToCenter / max(sizeX, sizeY, sizeZ)

        distanceToCenter = (
            distanceToCenter.unsqueeze(0).expand(n_labels, -1, -1, -1).to(self.device)
        )

        geodesic_perlin = torch.clamp(
            minmax_norm(perlin_noise) - distanceToCenter, 0, 1
        )

        return geodesic_perlin

    def _pad_label_maps(
        self, label_map: torch.Tensor, n_labels: int = 10
    ) -> torch.Tensor:
        """Pad a label map with zeros to match a target size on the first dimension.

        Args:
            label_map (torch.Tensor): The input tensor to pad.
            n_labels (int): The target size for the first dimension (default is 10).

        Returns:
            torch.Tensor: A tensor of shape padded with zeros to match the target size on the first dimension.
        """
        # Get the current shape of the input tensor
        labels_shape, *img_shape = label_map.shape

        if labels_shape < n_labels:
            # Fill with zeros to match the target size
            padding = torch.zeros((n_labels - labels_shape, *img_shape)).to(self.device)

            # Concatenate the input tensor with the padding along the first dimension
            padded_label_map = torch.cat((label_map, padding), dim=0)
        else:
            padded_label_map = label_map

        return padded_label_map

    def generate_shapes(self) -> tuple:
        """Generate a single 3D random shape.

        Returns:
            Tuple: The generated sinogram, image, CT image, label map tensors and the labels mapping dict.
        """

        # Generate the labels
        label_map_list = self._generate_label_maps()

        # Create the image from the labels
        image, ct_image, label_map, mapping = self._create_image_from_label_maps(label_map_list)

        # Apply Gaussian blur to the image
        image_blur = F.gaussian_blur(
            image,
            kernel_size=[self.kernel_size, self.kernel_size],
            sigma=[self.sigma, self.sigma],
        )

        # Project the ellipsoids to generate the sinogram
        sinogram = self.projector(image_blur)

        # Add Poisson noise to the sinogram
        if self.noise_interval[1] > 0:
            noise_level = np.random.uniform(*self.noise_interval)
            # Clamp the sinogram to be non-negative
            sinogram = torch.clamp_min(sinogram, 0)
            # Add Poisson noise
            sinogram = torch.poisson(sinogram / noise_level) * noise_level

        return (sinogram, ct_image, image, label_map, mapping)


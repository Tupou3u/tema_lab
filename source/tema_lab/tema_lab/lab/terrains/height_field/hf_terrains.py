# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Functions to generate height fields for different terrains."""

from __future__ import annotations

import numpy as np
from typing import TYPE_CHECKING

from isaaclab.terrains.height_field.utils import height_field_to_mesh

if TYPE_CHECKING:
    from . import hf_terrains_cfg

from perlin_noise import PerlinNoise


@height_field_to_mesh
def noisy_perlin_terrain(difficulty: float, cfg: hf_terrains_cfg.HfNoisyPerlinTerrainCfg) -> np.ndarray:
    """Faster version using precomputed coordinates."""
    
    # Resolve terrain configuration
    amplitude = cfg.amplitude_range[0] + difficulty * (cfg.amplitude_range[1] - cfg.amplitude_range[0])
    frequency = cfg.frequency_range[0] + difficulty * (cfg.frequency_range[1] - cfg.frequency_range[0])
    
    width_pixels = int(cfg.size[0] / cfg.horizontal_scale)
    length_pixels = int(cfg.size[1] / cfg.horizontal_scale)
    amplitude_pixels = int(amplitude / cfg.vertical_scale)
    
    # Create coordinate arrays
    x_coords = np.linspace(0, frequency, width_pixels)
    y_coords = np.linspace(0, frequency, length_pixels)
    
    # Initialize noise generator
    noise = PerlinNoise(
        octaves=cfg.octaves,
        seed=cfg.seed
    )
    
    # Generate noise using list comprehension (быстрее чем вложенные циклы)
    noise_map = np.array([[noise([x, y]) for y in y_coords] for x in x_coords])
    
    # Normalize and scale
    noise_map = (noise_map - np.min(noise_map)) / (np.max(noise_map) - np.min(noise_map))
    noise_map = 2 * noise_map - 1  # Convert to range [-1, 1]
    
    hf_raw = noise_map * amplitude_pixels

    if cfg.enable_noise:      
        # Convert noise parameters to discrete units
        noise_min_discrete = int(cfg.noise_range[0] / cfg.vertical_scale)
        noise_max_discrete = int(cfg.noise_range[1] / cfg.vertical_scale)
        noise_step_discrete = int(cfg.noise_step / cfg.vertical_scale)
        
        # Create random additional noise
        if noise_step_discrete > 0:
            noise_range_values = np.arange(noise_min_discrete, noise_max_discrete + noise_step_discrete, noise_step_discrete)
            additional_noise = np.random.choice(noise_range_values, size=(width_pixels, length_pixels))
        else:
            # Continuous noise if step is 0
            additional_noise = np.random.uniform(noise_min_discrete, noise_max_discrete, size=(width_pixels, length_pixels))
        
        # Add additional noise to the Perlin noise
        hf_raw = hf_raw + additional_noise
    
    return np.rint(hf_raw).astype(np.int16)


@height_field_to_mesh
def noisy_pyramid_sloped_terrain(difficulty: float, cfg: hf_terrains_cfg.HfNoisyPyramidSlopedTerrainCfg) -> np.ndarray:
    """Generate a terrain with a truncated pyramid structure with slope-dependent noise.

    The terrain is a pyramid-shaped sloped surface with a slope of :obj:`slope` that trims into a flat platform
    at the center, with added random noise whose intensity depends on the slope angle.

    Args:
        difficulty: The difficulty of the terrain. This is a value between 0 and 1.
        cfg: The configuration for the terrain.

    Returns:
        The height field of the terrain as a 2D numpy array with discretized heights.
    """
    # resolve terrain configuration
    if cfg.inverted:
        slope = -cfg.slope_range[0] - difficulty * (cfg.slope_range[1] - cfg.slope_range[0])
    else:
        slope = cfg.slope_range[0] + difficulty * (cfg.slope_range[1] - cfg.slope_range[0])

    # switch parameters to discrete units
    # -- horizontal scale
    width_pixels = int(cfg.size[0] / cfg.horizontal_scale)
    length_pixels = int(cfg.size[1] / cfg.horizontal_scale)
    # -- height
    # we want the height to be 1/2 of the width since the terrain is a pyramid
    height_max = int(slope * cfg.size[0] / 2 / cfg.vertical_scale)
    # -- center of the terrain
    center_x = int(width_pixels / 2)
    center_y = int(length_pixels / 2)

    # create a meshgrid of the terrain
    x = np.arange(0, width_pixels)
    y = np.arange(0, length_pixels)
    xx, yy = np.meshgrid(x, y, sparse=True)
    # offset the meshgrid to the center of the terrain
    xx = (center_x - np.abs(center_x - xx)) / center_x
    yy = (center_y - np.abs(center_y - yy)) / center_y
    # reshape the meshgrid to be 2D
    xx = xx.reshape(width_pixels, 1)
    yy = yy.reshape(1, length_pixels)
    # create a sloped surface
    hf_raw = np.zeros((width_pixels, length_pixels))
    hf_raw = height_max * xx * yy

    # create a flat platform at the center of the terrain
    platform_width = int(cfg.platform_width / cfg.horizontal_scale / 2)
    # get the height of the platform at the corner of the platform
    x_pf = width_pixels // 2 - platform_width
    y_pf = length_pixels // 2 - platform_width
    z_pf = hf_raw[x_pf, y_pf]
    hf_raw = np.clip(hf_raw, min(0, z_pf), max(0, z_pf))

    if cfg.enable_noise:

        # Calculate noise scaling factor based on slope (0-1 range)
        # Use absolute value of slope and normalize to maximum possible slope
        max_possible_slope = max(abs(cfg.slope_range[0]), abs(cfg.slope_range[1]))
        slope_factor = min(abs(slope) / max_possible_slope, 1.0)
        
        # Scale noise range based on slope factor
        # You can use linear or non-linear scaling - here's linear:
        noise_min_scaled = cfg.noise_range[0] * slope_factor
        noise_max_scaled = cfg.noise_range[1] * slope_factor
        
        # Convert scaled noise parameters to discrete units
        noise_min_discrete = int(noise_min_scaled / cfg.vertical_scale)
        noise_max_discrete = int(noise_max_scaled / cfg.vertical_scale)
        noise_step_discrete = int(cfg.noise_step / cfg.vertical_scale)
        
        # Create random noise scaled by slope factor
        if noise_step_discrete > 0:
            noise_range_values = np.arange(noise_min_discrete, noise_max_discrete + noise_step_discrete, noise_step_discrete)
            noise = np.random.choice(noise_range_values, size=(width_pixels, length_pixels))
        else:
            # Continuous noise if step is 0
            noise = np.random.uniform(noise_min_discrete, noise_max_discrete, size=(width_pixels, length_pixels))
        
        # Add noise to the base terrain
        hf_raw = hf_raw + noise
        
        # Re-clip to maintain platform flatness but allow noise
        hf_raw = np.clip(hf_raw, min(0, z_pf - noise_max_discrete), max(0, z_pf + noise_max_discrete))
    
    return np.rint(hf_raw).astype(np.int16)


@height_field_to_mesh
def noisy_wave_terrain(difficulty: float, cfg: hf_terrains_cfg.HfNoisyWaveTerrainCfg) -> np.ndarray:
    r"""Generate a terrain with a wave pattern with optional noise.

    The terrain is a flat platform at the center of the terrain with a wave pattern. The wave pattern
    is generated by adding sinusoidal waves based on the number of waves and the amplitude of the waves.

    The height of the terrain at a point :math:`(x, y)` is given by:

    .. math::

        h(x, y) =  A \left(\sin\left(\frac{2 \pi x}{\lambda}\right) + \cos\left(\frac{2 \pi y}{\lambda}\right) \right)

    where :math:`A` is the amplitude of the waves, :math:`\lambda` is the wavelength of the waves.

    .. image:: ../../_static/terrains/height_field/wave_terrain.jpg
       :width: 40%
       :align: center

    Args:
        difficulty: The difficulty of the terrain. This is a value between 0 and 1.
        cfg: The configuration for the terrain.

    Returns:
        The height field of the terrain as a 2D numpy array with discretized heights.
        The shape of the array is (width, length), where width and length are the number of points
        along the x and y axis, respectively.

    Raises:
        ValueError: When the number of waves is non-positive.
    """
    # check number of waves
    if cfg.num_waves < 0:
        raise ValueError(f"Number of waves must be a positive integer. Got: {cfg.num_waves}.")

    # resolve terrain configuration
    amplitude = cfg.amplitude_range[0] + difficulty * (cfg.amplitude_range[1] - cfg.amplitude_range[0])
    # switch parameters to discrete units
    # -- terrain
    width_pixels = int(cfg.size[0] / cfg.horizontal_scale)
    length_pixels = int(cfg.size[1] / cfg.horizontal_scale)
    amplitude_pixels = int(0.5 * amplitude / cfg.vertical_scale)

    # compute the wave number: nu = 2 * pi / lambda
    wave_length = length_pixels / cfg.num_waves
    wave_number = 2 * np.pi / wave_length
    # create meshgrid for the terrain
    x = np.arange(0, width_pixels)
    y = np.arange(0, length_pixels)
    xx, yy = np.meshgrid(x, y, sparse=True)
    xx = xx.reshape(width_pixels, 1)
    yy = yy.reshape(1, length_pixels)

    # create a terrain with a flat platform at the center
    hf_raw = np.zeros((width_pixels, length_pixels))
    # add the waves
    hf_raw += amplitude_pixels * (np.cos(yy * wave_number) + np.sin(xx * wave_number))
        
    if cfg.enable_noise:
        # Calculate noise scaling factor based on wave amplitude (0-1 range)
        # Use relative amplitude compared to maximum possible amplitude
        max_possible_amplitude = max(abs(cfg.amplitude_range[0]), abs(cfg.amplitude_range[1]))
        amplitude_factor = min(abs(amplitude) / max_possible_amplitude, 1.0)
        
        # Scale noise range based on amplitude factor
        noise_min_scaled = cfg.noise_range[0] * amplitude_factor
        noise_max_scaled = cfg.noise_range[1] * amplitude_factor
        
        # Convert scaled noise parameters to discrete units
        noise_min_discrete = int(noise_min_scaled / cfg.vertical_scale)
        noise_max_discrete = int(noise_max_scaled / cfg.vertical_scale)
        noise_step_discrete = int(cfg.noise_step / cfg.vertical_scale)
        
        # Create random noise scaled by amplitude factor
        if noise_step_discrete > 0:
            noise_range_values = np.arange(noise_min_discrete, noise_max_discrete + noise_step_discrete, noise_step_discrete)
            noise = np.random.choice(noise_range_values, size=(width_pixels, length_pixels))
        else:
            # Continuous noise if step is 0
            noise = np.random.uniform(noise_min_discrete, noise_max_discrete, size=(width_pixels, length_pixels))
        
        # Add noise to the base terrain
        hf_raw = hf_raw + noise
    
    # round off the heights to the nearest vertical step
    return np.rint(hf_raw).astype(np.int16)


@height_field_to_mesh
def noisy_discrete_obstacles_terrain(difficulty: float, cfg: hf_terrains_cfg.HfNoisyDiscreteObstaclesTerrainCfg) -> np.ndarray:
    """Generate a terrain with randomly generated obstacles as pillars with positive and negative heights.

    The terrain is a flat platform at the center of the terrain with randomly generated obstacles as pillars
    with positive and negative height. The obstacles are randomly generated cuboids with a random width and
    height. They are placed randomly on the terrain with a minimum distance of :obj:`cfg.platform_width`
    from the center of the terrain.

    .. image:: ../../_static/terrains/height_field/discrete_obstacles_terrain.jpg
       :width: 40%
       :align: center

    Args:
        difficulty: The difficulty of the terrain. This is a value between 0 and 1.
        cfg: The configuration for the terrain.

    Returns:
        The height field of the terrain as a 2D numpy array with discretized heights.
        The shape of the array is (width, length), where width and length are the number of points
        along the x and y axis, respectively.
    """
    # resolve terrain configuration
    obs_height = cfg.obstacle_height_range[0] + difficulty * (
        cfg.obstacle_height_range[1] - cfg.obstacle_height_range[0]
    )

    # switch parameters to discrete units
    # -- terrain
    width_pixels = int(cfg.size[0] / cfg.horizontal_scale)
    length_pixels = int(cfg.size[1] / cfg.horizontal_scale)
    # -- obstacles
    obs_height = int(obs_height / cfg.vertical_scale)
    obs_width_min = int(cfg.obstacle_width_range[0] / cfg.horizontal_scale)
    obs_width_max = int(cfg.obstacle_width_range[1] / cfg.horizontal_scale)
    # -- center of the terrain
    platform_width = int(cfg.platform_width / cfg.horizontal_scale)

    # create discrete ranges for the obstacles
    # -- shape
    obs_width_range = np.arange(obs_width_min, obs_width_max, 4)
    obs_length_range = np.arange(obs_width_min, obs_width_max, 4)
    # -- position
    obs_x_range = np.arange(0, width_pixels, 4)
    obs_y_range = np.arange(0, length_pixels, 4)

    # create a terrain with a flat platform at the center
    hf_raw = np.zeros((width_pixels, length_pixels))
    # generate the obstacles
    for _ in range(cfg.num_obstacles):
        # sample size
        if cfg.obstacle_height_mode == "choice":
            height = np.random.choice([-obs_height, -obs_height // 2, obs_height // 2, obs_height])
        elif cfg.obstacle_height_mode == "fixed":
            height = obs_height
        else:
            raise ValueError(f"Unknown obstacle height mode '{cfg.obstacle_height_mode}'. Must be 'choice' or 'fixed'.")
        width = int(np.random.choice(obs_width_range))
        length = int(np.random.choice(obs_length_range))
        # sample position
        x_start = int(np.random.choice(obs_x_range))
        y_start = int(np.random.choice(obs_y_range))
        # clip start position to the terrain
        if x_start + width > width_pixels:
            x_start = width_pixels - width
        if y_start + length > length_pixels:
            y_start = length_pixels - length
        # add to terrain
        hf_raw[x_start : x_start + width, y_start : y_start + length] = height
    # clip the terrain to the platform
    x1 = (width_pixels - platform_width) // 2
    x2 = (width_pixels + platform_width) // 2
    y1 = (length_pixels - platform_width) // 2
    y2 = (length_pixels + platform_width) // 2
    hf_raw[x1:x2, y1:y2] = 0
        
    if cfg.enable_noise:        
        # Convert noise parameters to discrete units
        noise_min_discrete = int(cfg.noise_range[0] / cfg.vertical_scale)
        noise_max_discrete = int(cfg.noise_range[1] / cfg.vertical_scale)
        noise_step_discrete = int(cfg.noise_step / cfg.vertical_scale)
        
        # Create random noise
        if noise_step_discrete > 0:
            noise_range_values = np.arange(noise_min_discrete, noise_max_discrete + noise_step_discrete, noise_step_discrete)
            noise = np.random.choice(noise_range_values, size=(width_pixels, length_pixels))
        else:
            # Continuous noise if step is 0
            noise = np.random.uniform(noise_min_discrete, noise_max_discrete, size=(width_pixels, length_pixels))
        
        # Add noise to the base terrain, but preserve the flat platform
        hf_raw = hf_raw + noise
        
    # return the original terrain without noise
    return np.rint(hf_raw).astype(np.int16)
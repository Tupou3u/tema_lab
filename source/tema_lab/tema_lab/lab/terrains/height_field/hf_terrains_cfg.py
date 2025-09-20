# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import MISSING
from isaaclab.utils import configclass
from . import hf_terrains
from isaaclab.terrains.height_field.hf_terrains_cfg import HfTerrainBaseCfg, HfPyramidSlopedTerrainCfg, HfWaveTerrainCfg, HfDiscreteObstaclesTerrainCfg


@configclass
class HfNoisyPerlinTerrainCfg(HfTerrainBaseCfg):
    function = hf_terrains.noisy_perlin_terrain
    amplitude_range: tuple[float, float] = MISSING
    frequency_range: tuple[float, float] = MISSING
    octaves: int = MISSING
    persistence: float = MISSING
    lacunarity: float = MISSING
    seed: int = 42
    enable_noise: bool = True
    noise_range: tuple[float, float] = (0.01, 0.05) 
    noise_step: float = 0.01 


@configclass
class HfNoisyPyramidSlopedTerrainCfg(HfPyramidSlopedTerrainCfg):
    function = hf_terrains.noisy_pyramid_sloped_terrain
    enable_noise: bool = True
    noise_range: tuple[float, float] = (0.01, 0.05) 
    noise_step: float = 0.01 


@configclass
class HfNoisyWaveTerrainCfg(HfWaveTerrainCfg):
    function = hf_terrains.noisy_wave_terrain
    enable_noise: bool = True
    noise_range: tuple[float, float] = (0.01, 0.05) 
    noise_step: float = 0.01 


@configclass
class HfNoisyDiscreteObstaclesTerrainCfg(HfDiscreteObstaclesTerrainCfg):
    function = hf_terrains.noisy_discrete_obstacles_terrain
    enable_noise: bool = True
    noise_range: tuple[float, float] = (0.01, 0.05) 
    noise_step: float = 0.01 

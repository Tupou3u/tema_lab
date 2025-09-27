# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import isaaclab.terrains as terrain_gen
from isaaclab.terrains.terrain_generator_cfg import TerrainGeneratorCfg

BASE_ROUGH_TERRAINS_CFG = TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=50.0,
    num_rows=10,
    num_cols=20,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    sub_terrains={
        "flat": terrain_gen.MeshPlaneTerrainCfg(
            proportion=1.0
        ),
        "pyramid_stairs": terrain_gen.MeshPyramidStairsTerrainCfg(
            proportion=0.5,
            step_height_range=(0.05, 0.2),
            step_width=0.3,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
        ),
        "pyramid_stairs_inv": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
            proportion=0.5,
            step_height_range=(0.05, 0.2),
            step_width=0.3,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
        ),
        "boxes": terrain_gen.MeshRandomGridTerrainCfg(
            proportion=1.0, grid_width=0.45, grid_height_range=(0.01, 0.1), platform_width=2.0
        ),
        "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=1.0, noise_range=(0.01, 0.1), noise_step=0.01, border_width=0.25
        ),
        "hf_pyramid_slope": terrain_gen.HfPyramidSlopedTerrainCfg(
            proportion=0.5, slope_range=(0.0, 0.3), platform_width=2.0, border_width=0.25
        ),
        "hf_pyramid_slope_inv": terrain_gen.HfInvertedPyramidSlopedTerrainCfg(
            proportion=0.5, slope_range=(0.0, 0.3), platform_width=2.0, border_width=0.25
        )
    },
)
"""Rough terrains configuration."""


HARD_ROUGH_TERRAINS_CFG = TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=50.0,
    num_rows=10,
    num_cols=84,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    sub_terrains={
        # Mesh
        "flat": terrain_gen.MeshPlaneTerrainCfg(proportion=1.0),
        "pyramid_stairs_0.1": terrain_gen.MeshPyramidStairsTerrainCfg(
            proportion=0.16,
            step_height_range=(0.05, 0.1),
            step_width=0.2,
            border_width=1.0,
            platform_width=2.0
        ),
        "pyramid_stairs_0.1_inv": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
            proportion=0.16,
            step_height_range=(0.05, 0.1),
            step_width=0.2,
            border_width=1.0,
            platform_width=2.0
        ),
        "pyramid_stairs_0.2": terrain_gen.MeshPyramidStairsTerrainCfg(
            proportion=0.16,
            step_height_range=(0.05, 0.15),
            step_width=0.2,
            border_width=1.0,
            platform_width=2.0
        ),
        "pyramid_stairs_0.2_inv": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
            proportion=0.16,
            step_height_range=(0.05, 0.15),
            step_width=0.2,
            border_width=1.0,
            platform_width=2.0
        ),
        "pyramid_stairs_0.3": terrain_gen.MeshPyramidStairsTerrainCfg(
            proportion=0.16,
            step_height_range=(0.05, 0.2),
            step_width=0.3,
            border_width=1.0,
            platform_width=2.0
        ),
        "pyramid_stairs_0.3_inv": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
            proportion=0.16,
            step_height_range=(0.05, 0.2),
            step_width=0.3,
            border_width=1.0,
            platform_width=2.0
        ),
        "boxes_0.45": terrain_gen.MeshRandomGridTerrainCfg(
            proportion=0.33,
            grid_width=0.45, 
            grid_height_range=(0.025, 0.075)
        ),
        "boxes_0.3": terrain_gen.MeshRandomGridTerrainCfg(
            proportion=0.33,
            grid_width=0.3, 
            grid_height_range=(0.025, 0.075)
        ),
        "boxes_0.15": terrain_gen.MeshRandomGridTerrainCfg(
            proportion=0.33,
            grid_width=0.15, 
            grid_height_range=(0.025, 0.075)
        ),
        "repeat_boxes": terrain_gen.MeshRepeatedBoxesTerrainCfg(
            proportion=1.0,
            object_params_start=terrain_gen.MeshRepeatedBoxesTerrainCfg.ObjectCfg(
                num_objects=50, height=0.05, size=(0.1, 0.1), max_yx_angle=0.0
            ),
            object_params_end=terrain_gen.MeshRepeatedBoxesTerrainCfg.ObjectCfg(
                num_objects=50, height=0.05, size=(0.5, 0.5), max_yx_angle=30.0
            ),
            platform_width=2.0,
        ),
        "star": terrain_gen.MeshStarTerrainCfg(
            proportion=1.0,
            num_bars=10, 
            bar_width_range=(0.1, 0.2), 
            bar_height_range=(0.05, 0.2)
        ),
        # Hf
        "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=1.0,
            noise_range=(0.01, 0.1), 
            noise_step=0.01, 
            border_width=0.25
        ),
        "pyramid_slope": terrain_gen.HfPyramidSlopedTerrainCfg(
            proportion=0.5,
            slope_range=(0.1, 0.4), 
            border_width=0.25
        ),
        "pyramid_slope_inv": terrain_gen.HfInvertedPyramidSlopedTerrainCfg(
            proportion=0.5,
            slope_range=(0.1, 0.4), 
            border_width=0.25
        ),
        "noisy_pyramid_slope": terrain_gen.HfNoisyPyramidSlopedTerrainCfg(
            proportion=0.5,
            slope_range=(0.1, 0.4),
            noise_range=(0.01, 0.05), 
            noise_step=0.01,
            border_width=0.25
        ),
        "noisy_pyramid_slope_inv": terrain_gen.HfNoisyPyramidSlopedTerrainCfg(
            proportion=0.5,
            slope_range=(0.1, 0.4),
            noise_range=(0.01, 0.05), 
            noise_step=0.01,
            inverted=True,
            border_width=0.25
        ),
        "wave": terrain_gen.HfWaveTerrainCfg(
            proportion=1.0,
            amplitude_range=(0.0, 0.2), 
            num_waves=5, 
            border_width=0.25
        ),
        "noisy_wave": terrain_gen.HfNoisyWaveTerrainCfg(
            proportion=1.0,
            amplitude_range=(0.0, 0.2), 
            num_waves=5, 
            noise_range=(0.01, 0.05), 
            noise_step=0.01,
            border_width=0.25
        ),
        "discrete_obtacles": terrain_gen.HfDiscreteObstaclesTerrainCfg(
            proportion=1.0,
            border_width=0.25, 
            obstacle_height_range=(0.025, 0.1),
            obstacle_width_range=(0.05, 1.0),
            num_obstacles=200
        ),
        "noisy_discrete_obtacles": terrain_gen.HfNoisyDiscreteObstaclesTerrainCfg(
            proportion=1.0,
            border_width=0.25, 
            obstacle_height_range=(0.025, 0.1),
            obstacle_width_range=(0.05, 1.0),
            num_obstacles=200,
            noise_range=(0.01, 0.05), 
            noise_step=0.01
        ),
        "perlin_terrain": terrain_gen.HfNoisyPerlinTerrainCfg(
            proportion=1.0,
            amplitude_range=(0.1, 0.3),
            frequency_range=(0.5, 1.5),
            octaves=4,
            persistence=0.1,
            lacunarity=2,
            enable_noise=False,
            border_width=0.25,
        ),
        "noisy_perlin_terrain": terrain_gen.HfNoisyPerlinTerrainCfg(
            proportion=1.0,
            amplitude_range=(0.1, 0.3),
            frequency_range=(0.5, 1.5),
            octaves=4,
            persistence=0.1,
            lacunarity=2,
            noise_range=(0.01, 0.05), 
            noise_step=0.01,
            border_width=0.25
        ),
    },
)


EASY_ROUGH_TERRAINS_CFG = TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=50.0,
    num_rows=10,
    num_cols=20,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    sub_terrains={
        # Mesh
        "flat": terrain_gen.MeshPlaneTerrainCfg(proportion=1.0),
        # Hf
        "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=1.0,
            noise_range=(0.01, 0.08), 
            noise_step=0.01, 
            border_width=0.25
        ),
        "boxes_0.45": terrain_gen.MeshRandomGridTerrainCfg(
            proportion=1.0,
            grid_width=0.45, 
            grid_height_range=(0.02, 0.04)
        ),
        "pyramid_slope": terrain_gen.HfPyramidSlopedTerrainCfg(
            proportion=0.5,
            slope_range=(0.0, 0.3), 
            border_width=0.25
        ),
        "pyramid_slope_inv": terrain_gen.HfInvertedPyramidSlopedTerrainCfg(
            proportion=0.5,
            slope_range=(0.0, 0.3), 
            border_width=0.25
        ),
        "wave": terrain_gen.HfWaveTerrainCfg(
            proportion=1.0,
            amplitude_range=(0.0, 0.1), 
            num_waves=5, 
            border_width=0.25
        )
    },
)
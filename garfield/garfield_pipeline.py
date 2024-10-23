import typing
from dataclasses import dataclass, field
from typing import Literal, Type, Mapping, Any

import torch
from pathlib import Path
from nerfstudio.pipelines.base_pipeline import VanillaPipeline, VanillaPipelineConfig
from torch.cuda.amp.grad_scaler import GradScaler
from nerfstudio.viewer.viewer_elements import *
from nerfstudio.viewer.viewer import VISER_NERFSTUDIO_SCALE_RATIO
from nerfstudio.cameras.rays import RayBundle

import tqdm

from sklearn.preprocessing import QuantileTransformer
from garfield.garfield_datamanager import GarfieldDataManagerConfig, GarfieldDataManager
from garfield.garfield_model import GarfieldModel, GarfieldModelConfig


@dataclass
class GarfieldPipelineConfig(VanillaPipelineConfig):
    """Configuration for GARField pipeline instantiation"""

    _target: Type = field(default_factory=lambda: GarfieldPipeline)
    """target class to instantiate"""

    datamanager: GarfieldDataManagerConfig = field(default_factory=lambda: GarfieldDataManagerConfig())
    model: GarfieldModelConfig = field(default_factory=lambda: GarfieldModelConfig())

    start_grouping_step: int = 2000
    max_grouping_scale: float = 2.0
    num_rays_per_image: int = 256
    normalize_grouping_scale: bool = True


class GarfieldPipeline(VanillaPipeline):
    config: GarfieldPipelineConfig
    datamanager: GarfieldDataManager
    model: GarfieldModel

    def __init__(
        self,
        config: GarfieldPipelineConfig,
        device: str,
        test_mode: Literal["test", "val", "inference"] = "val",
        world_size: int = 1,
        local_rank: int = 0,
        grad_scaler: typing.Optional[GradScaler] = None,
    ):
        config.model.max_grouping_scale = config.max_grouping_scale
        super().__init__(
            config,
            device,
            test_mode,
            world_size,
            local_rank,
            grad_scaler,
        )

        self.z_export_options = ViewerCheckbox(name="Export Options", default_value=False, cb_hook=self._update_export_options)
        self.z_export_similar_affinities = ViewerButton(
            name="Export Similar Affinity Pointcloud",
            visible=False,
            cb_hook=self._export_similar_affinities
        )

    def _update_export_options(self, checkbox: ViewerCheckbox):
        """Update the UI based on the export options"""
        self.z_export_similar_affinities.set_hidden(not checkbox.value)

    def get_train_loss_dict(self, step: int):
        """In addition to the base class, we also calculate SAM masks
        and their 3D scales at `start_grouping_step`."""
        if step == self.config.start_grouping_step:
            loaded = self.datamanager.load_sam_data()
            if not loaded:
                self.populate_grouping_info()
            else:
                # Initialize grouping statistics. This will be automatically loaded from a checkpoint next time.
                scale_stats = self.datamanager.scale_3d_statistics
                self.grouping_stats = torch.nn.Parameter(scale_stats)
                self.model.grouping_field.quantile_transformer = (
                    self._get_quantile_func(scale_stats)
                )
            # Set the number of rays per image to the number of rays per image for grouping
            pixel_sampler = self.datamanager.train_pixel_sampler
            pixel_sampler.num_rays_per_image = pixel_sampler.config.num_rays_per_image

        ray_bundle, batch = self.datamanager.next_train(step)
        if step >= self.config.start_grouping_step:
            # also set the grouping info in the batch; in-place operation
            self.datamanager.next_group(ray_bundle, batch)

        model_outputs = self._model(
            ray_bundle
        )  # train distributed data parallel model if world_size > 1

        metrics_dict = self.model.get_metrics_dict(model_outputs, batch)
        loss_dict = self.model.get_loss_dict(model_outputs, batch, metrics_dict)
        if step >= self.config.start_grouping_step:
            loss_dict.update(
                self.model.get_loss_dict_group(model_outputs, batch, metrics_dict)
            )

        return model_outputs, loss_dict, metrics_dict

    def populate_grouping_info(self):
        """
        Calculate groups from SAM and their 3D scales, and save them in the datamanager.
        This information is required to supervise the grouping field.
        """
        # Note that pipeline is in train mode here, via the base trainer.
        self.model.eval()

        # Calculate multi-scale masks, and their 3D scales
        scales_3d_list, pixel_level_keys_list, group_cdf_list = [], [], []
        train_cameras = self.datamanager.train_dataset.cameras
        for i in tqdm.trange(len(train_cameras), desc="Calculating 3D masks"):
            camera_ray_bundle = train_cameras.generate_rays(camera_indices=i).to(
                self.device
            )
            with torch.no_grad():
                outputs = self.model.get_outputs_for_camera_ray_bundle(
                    camera_ray_bundle
                )

            # Get RGB (for SAM mask generation), depth and 3D point locations (for 3D scale calculation)
            rgb = self.datamanager.train_dataset[i]["image"]
            depth = outputs["depth"]
            points = camera_ray_bundle.origins + camera_ray_bundle.directions * depth
            # Scales are capped to `max_grouping_scale` to filter noisy / outlier masks.
            (
                pixel_level_keys,
                scale_3d,
                group_cdf,
            ) = self.datamanager._calculate_3d_groups(
                rgb, depth, points, max_scale=self.config.max_grouping_scale
            )

            pixel_level_keys_list.append(pixel_level_keys)
            scales_3d_list.append(scale_3d)
            group_cdf_list.append(group_cdf)

        # Save grouping data, and set it in the datamanager for current training.
        # This will be cached, so we don't need to calculate it again.
        self.datamanager.save_sam_data(
            pixel_level_keys_list, scales_3d_list, group_cdf_list
        )
        self.datamanager.pixel_level_keys = torch.nested.nested_tensor(
            pixel_level_keys_list
        )
        self.datamanager.scale_3d = torch.nested.nested_tensor(scales_3d_list)
        self.datamanager.group_cdf = torch.nested.nested_tensor(group_cdf_list)

        # Initialize grouping statistics. This will be automatically loaded from a checkpoint next time.
        self.grouping_stats = torch.nn.Parameter(torch.cat(scales_3d_list))
        self.model.grouping_field.quantile_transformer = self._get_quantile_func(
            torch.cat(scales_3d_list)
        )

        # Turn model back to train mode
        self.model.train()

    def load_state_dict(self, state_dict: Mapping[str, Any], strict: bool = True):
        """
        Same as the base class, but also loads the grouping statistics.
        It's important to normalize the 3D scales as input to the grouping field.
        """
        # Load 3D group scale statistics
        grouping_stats = state_dict["grouping_stats"]
        self.grouping_stats = torch.nn.Parameter(torch.zeros_like(grouping_stats)).to(
            self.device
        )
        # Calculate quantile transformer
        self.model.grouping_field.quantile_transformer = self._get_quantile_func(
            grouping_stats
        )

        return super().load_state_dict(state_dict, strict)

    def _get_quantile_func(self, scales: torch.Tensor, distribution="normal"):
        """
        Use 3D scale statistics to normalize scales -- use quantile transformer.
        """
        scales = scales.flatten()
        scales = scales[(scales > 0) & (scales < self.config.max_grouping_scale)]

        scales = scales.detach().cpu().numpy()

        # Calculate quantile transformer
        quantile_transformer = QuantileTransformer(output_distribution=distribution)
        quantile_transformer = quantile_transformer.fit(scales.reshape(-1, 1))

        def quantile_transformer_func(scales):
            # This function acts as a wrapper for QuantileTransformer.
            # QuantileTransformer expects a numpy array, while we have a torch tensor.
            return torch.Tensor(
                quantile_transformer.transform(scales.cpu().numpy())
            ).to(scales.device)

        return quantile_transformer_func


    def _export_similar_affinities(self, button: ViewerButton):
        """Export the similar affinities to a .ply file"""

        # location to save
        output_dir = f"outputs/{self.datamanager.config.dataparser.data.name}"
        filename = Path(output_dir) / f"pointcloud.ply"
        model = self.model

        # Whether the normals should be estimated based on the point cloud.
        
        num_points = 1000000
        remove_outliers = True
        estimate_normals = False
        reorient_normals = False
        rgb_output_name= "rgb"
        depth_output_name = "depth"
        normal_output_name = None
        crop_obb = None
        std_ratio = 10.0
        save_world_frame = False

        from rich.progress import BarColumn, Progress, TaskProgressColumn, TextColumn, TimeRemainingColumn
        progress = Progress(
            TextColumn(":cloud: Computing Point Cloud :cloud:"),
            BarColumn(),
            TaskProgressColumn(show_speed=True),
            TimeRemainingColumn(elapsed_when_finished=True, compact=True),
            console=CONSOLE,
        )
        points = []
        rgbs = []
        normals = []
        view_directions = []

        with progress as progress_bar:
            task = progress_bar.add_task("Generating Point Cloud", total=num_points)
            while not progress_bar.finished:
                normal = None
            
                with torch.no_grad():
                    ray_bundle, _ = self.datamanager.next_train(0)
                    assert isinstance(ray_bundle, RayBundle)
                    outputs = self.model(ray_bundle)
                if rgb_output_name not in outputs:
                    CONSOLE.rule("Error", style="red")
                    CONSOLE.print(f"Could not find {rgb_output_name} in the model outputs", justify="center")
                    CONSOLE.print(f"Please set --rgb_output_name to one of: {outputs.keys()}", justify="center")
                    sys.exit(1)
                if depth_output_name not in outputs:
                    CONSOLE.rule("Error", style="red")
                    CONSOLE.print(f"Could not find {depth_output_name} in the model outputs", justify="center")
                    CONSOLE.print(f"Please set --depth_output_name to one of: {outputs.keys()}", justify="center")
                    sys.exit(1)
                rgba = self.model.get_rgba_image(outputs, rgb_output_name)
                depth = outputs[depth_output_name]
                if normal_output_name is not None:
                    if normal_output_name not in outputs:
                        CONSOLE.rule("Error", style="red")
                        CONSOLE.print(f"Could not find {normal_output_name} in the model outputs", justify="center")
                        CONSOLE.print(f"Please set --normal_output_name to one of: {outputs.keys()}", justify="center")
                        sys.exit(1)
                    normal = outputs[normal_output_name]
                    assert (
                        torch.min(normal) >= 0.0 and torch.max(normal) <= 1.0
                    ), "Normal values from method output must be in [0, 1]"
                    normal = (normal * 2.0) - 1.0
                point = ray_bundle.origins + ray_bundle.directions * depth
                view_direction = ray_bundle.directions

                # Filter points with opacity lower than 0.5
                mask = rgba[..., -1] > 0.5
                point = point[mask]
                view_direction = view_direction[mask]
                rgb = rgba[mask][..., :3]
                if normal is not None:
                    normal = normal[mask]

                if crop_obb is not None:
                    mask = crop_obb.within(point)
                    point = point[mask]
                    rgb = rgb[mask]
                    view_direction = view_direction[mask]
                    if normal is not None:
                        normal = normal[mask]

                points.append(point)
                rgbs.append(rgb)
                view_directions.append(view_direction)
                if normal is not None:
                    normals.append(normal)
                progress.advance(task, point.shape[0])
        points = torch.cat(points, dim=0)
        rgbs = torch.cat(rgbs, dim=0)
        view_directions = torch.cat(view_directions, dim=0).cpu()

        import open3d as o3d
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points.double().cpu().numpy())
        pcd.colors = o3d.utility.Vector3dVector(rgbs.double().cpu().numpy())

        ind = None
        if remove_outliers:
            CONSOLE.print("Cleaning Point Cloud")
            pcd, ind = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=std_ratio)
            print("\033[A\033[A")
            CONSOLE.print("[bold green]:white_check_mark: Cleaning Point Cloud")
            if ind is not None:
                view_directions = view_directions[ind]

        # either estimate_normals or normal_output_name, not both
        if estimate_normals:
            if normal_output_name is not None:
                CONSOLE.rule("Error", style="red")
                CONSOLE.print("Cannot estimate normals and use normal_output_name at the same time", justify="center")
                sys.exit(1)
            CONSOLE.print("Estimating Point Cloud Normals")
            pcd.estimate_normals()
            print("\033[A\033[A")
            CONSOLE.print("[bold green]:white_check_mark: Estimating Point Cloud Normals")
        elif normal_output_name is not None:
            normals = torch.cat(normals, dim=0)
            if ind is not None:
                # mask out normals for points that were removed with remove_outliers
                normals = normals[ind]
            pcd.normals = o3d.utility.Vector3dVector(normals.double().cpu().numpy())

        # re-orient the normals
        if reorient_normals:
            normals = torch.from_numpy(np.array(pcd.normals)).float()
            mask = torch.sum(view_directions * normals, dim=-1) > 0
            normals[mask] *= -1
            pcd.normals = o3d.utility.Vector3dVector(normals.double().cpu().numpy())

        
        if save_world_frame:
            # apply the inverse dataparser transform to the point cloud
            points = np.asarray(pcd.points)
            poses = np.eye(4, dtype=np.float32)[None, ...].repeat(points.shape[0], axis=0)[:, :3, :]
            poses[:, :3, 3] = points
            poses = pipeline.datamanager.train_dataparser_outputs.transform_poses_to_original_space(
                torch.from_numpy(poses)
            )
            points = poses[:, :3, 3].numpy()
            pcd.points = o3d.utility.Vector3dVector(points)

        torch.cuda.empty_cache()

        CONSOLE.print(f"[bold green]:white_check_mark: Generated {pcd}")
        CONSOLE.print("Saving Point Cloud...")
        tpcd = o3d.t.geometry.PointCloud.from_legacy(pcd)
        # The legacy PLY writer converts colors to UInt8,
        # let us do the same to save space.
        tpcd.point.colors = (tpcd.point.colors * 255).to(o3d.core.Dtype.UInt8)  # type: ignore
        o3d.t.io.write_point_cloud(str(filename), tpcd)
        print("\033[A\033[A")
        CONSOLE.print("[bold green]:white_check_mark: Saving Point Cloud to " + str(filename))


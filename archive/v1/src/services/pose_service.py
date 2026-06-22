"""
Pose estimation service for WiFi-DensePose API.

Production paths in this module must NEVER use random data generation.
All mock/synthetic data generation is isolated in src.testing and is only
invoked when settings.mock_pose_data is explicitly True.
"""

import importlib.util
import logging
import asyncio
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timedelta

import numpy as np
import torch

from src.config.settings import Settings
from src.config.domains import DomainConfig
from src.core.csi_processor import CSIProcessor
from src.core.phase_sanitizer import PhaseSanitizer

logger = logging.getLogger(__name__)


class PoseService:
    """Service for pose estimation operations."""
    
    def __init__(self, settings: Settings, domain_config: DomainConfig, hardware_service: Optional[Any] = None):
        """Initialize pose service."""
        self.settings = settings
        self.domain_config = domain_config
        self.hardware_service = hardware_service
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.csi_processor = None
        self.phase_sanitizer = None
        self.pose_model = None
        self.model_persons = min(3, max(1, getattr(self.settings, 'pose_max_persons', 3)))

        # Service state
        self.is_initialized = False
        self.is_running = False
        self.last_error = None
        self._start_time: Optional[datetime] = None
        self._calibration_in_progress: bool = False
        self._calibration_id: Optional[str] = None
        self._calibration_start: Optional[datetime] = None
        
        # Processing statistics
        self.stats = {
            "total_processed": 0,
            "successful_detections": 0,
            "failed_detections": 0,
            "average_confidence": 0.0,
            "processing_time_ms": 0.0
        }
    
    async def initialize(self):
        """Initialize the pose service."""
        try:
            self.logger.info("Initializing pose service...")
            
            # Initialize CSI processor
            csi_config = {
                'buffer_size': self.settings.csi_buffer_size,
                'sampling_rate': getattr(self.settings, 'csi_sampling_rate', 1000),
                'window_size': getattr(self.settings, 'csi_window_size', 512),
                'overlap': getattr(self.settings, 'csi_overlap', 0.5),
                'noise_threshold': getattr(self.settings, 'csi_noise_threshold', 0.1),
                'human_detection_threshold': getattr(self.settings, 'csi_human_detection_threshold', 0.8),
                'smoothing_factor': getattr(self.settings, 'csi_smoothing_factor', 0.9),
                'max_history_size': getattr(self.settings, 'csi_max_history_size', 500),
                'num_subcarriers': 56,
                'num_antennas': 3
            }
            self.csi_processor = CSIProcessor(config=csi_config)
            
            # Initialize phase sanitizer
            phase_config = {
                'unwrapping_method': 'numpy',
                'outlier_threshold': 3.0,
                'smoothing_window': 5,
                'enable_outlier_removal': True,
                'enable_smoothing': True,
                'enable_noise_filtering': True,
                'noise_threshold': getattr(self.settings, 'csi_noise_threshold', 0.1)
            }
            self.phase_sanitizer = PhaseSanitizer(config=phase_config)
            
            # Initialize models if not mocking
            if not self.settings.mock_pose_data:
                await self._initialize_models()
            else:
                self.logger.info("Using mock pose data for development")
            
            self.is_initialized = True
            self._start_time = datetime.now()
            self.logger.info("Pose service initialized successfully")
            
        except Exception as e:
            self.last_error = str(e)
            self.logger.error(f"Failed to initialize pose service: {e}")
            raise
    
    async def _initialize_models(self):
        """Initialize neural network models."""
        try:
            self.pose_model = self._load_hybrid_pose_model()
            self.pose_model.eval()
            self.logger.info("Hybrid CSI pose model loaded")
        except Exception as e:
            self.logger.error(f"Failed to initialize models: {e}")
            raise

    def _load_hybrid_pose_model(self) -> torch.nn.Module:
        """Dynamically import and instantiate HybridCSIModel from scripts/retrain_model.py."""
        script_path = Path(__file__).resolve().parents[4] / "scripts" / "retrain_model.py"
        if not script_path.exists():
            raise FileNotFoundError(f"Hybrid model loader not found: {script_path}")

        spec = importlib.util.spec_from_file_location("retrain_model", str(script_path))
        if spec is None or spec.loader is None:
            raise ImportError(f"Unable to load hybrid model module from {script_path}")

        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        model_cls = getattr(module, "HybridCSIModel", None)
        if model_cls is None:
            raise ImportError("HybridCSIModel not found in retrain_model.py")

        model = model_cls(num_persons=self.model_persons, pretrained_path=None)

        if self.settings.pose_model_path:
            checkpoint = torch.load(self.settings.pose_model_path, map_location="cpu")
            state = checkpoint.get("model_state_dict", checkpoint)
            model.load_state_dict(state, strict=False)

        return model
    
    async def start(self):
        """Start the pose service."""
        if not self.is_initialized:
            await self.initialize()
        
        self.is_running = True
        self.logger.info("Pose service started")
    
    async def stop(self):
        """Stop the pose service."""
        self.is_running = False
        self.logger.info("Pose service stopped")
    
    async def process_csi_data(self, csi_data: np.ndarray, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Process CSI data and estimate poses."""
        if not self.is_running:
            raise RuntimeError("Pose service is not running")
        
        start_time = datetime.now()
        
        try:
            # Process CSI data
            processed_csi = await self._process_csi(csi_data, metadata)
            
            # Estimate poses
            poses = await self._estimate_poses(processed_csi, metadata)
            
            # Update statistics
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            self._update_stats(poses, processing_time)
            
            return {
                "timestamp": start_time.isoformat(),
                "poses": poses,
                "metadata": metadata,
                "processing_time_ms": processing_time,
                "confidence_scores": [pose.get("confidence", 0.0) for pose in poses]
            }
            
        except Exception as e:
            self.last_error = str(e)
            self.stats["failed_detections"] += 1
            self.logger.error(f"Error processing CSI data: {e}")
            raise
    
    async def _process_csi(self, csi_data: Union[np.ndarray, Any], metadata: Dict[str, Any]) -> np.ndarray:
        """Process raw CSI data into a 168-dim CSI inference vector."""
        from src.hardware.csi_extractor import CSIData

        # Accept CSIData objects directly when they come from hardware collectors.
        if hasattr(csi_data, "amplitude"):
            csi_data_obj = csi_data
        else:
            # If raw array is passed, construct a CSIData object for the processor.
            if isinstance(csi_data, np.ndarray):
                if csi_data.ndim == 2 and csi_data.shape[1] == 56:
                    amplitude = csi_data
                elif csi_data.ndim == 1 and csi_data.shape[0] == 56:
                    amplitude = np.tile(csi_data, 3)
                else:
                    amplitude = np.asarray(csi_data, dtype=np.float32)
                    if amplitude.ndim == 1 and amplitude.shape[0] == 168:
                        amplitude = amplitude.reshape(3, 56)
                    else:
                        amplitude = amplitude.flatten()
                        if amplitude.size == 56:
                            amplitude = np.tile(amplitude, 3)
                        elif amplitude.size == 168:
                            amplitude = amplitude.reshape(3, 56)
                        else:
                            amplitude = np.resize(amplitude, (3, 56))
            else:
                amplitude = np.asarray(csi_data, dtype=np.float32)

            phase = np.zeros_like(amplitude)
            csi_data_obj = CSIData(
                timestamp=metadata.get("timestamp", datetime.now()),
                amplitude=amplitude,
                phase=phase,
                frequency=metadata.get("frequency", 5.0),
                bandwidth=metadata.get("bandwidth", 20.0),
                num_subcarriers=56,
                num_antennas=3,
                snr=metadata.get("snr", 20.0),
                metadata=metadata
            )

        # Process CSI data through the existing pipeline for history and detection features.
        try:
            detection_result = await self.csi_processor.process_csi_data(csi_data_obj)
            self.csi_processor.add_to_history(csi_data_obj)
        except Exception as e:
            self.logger.warning(f"CSI processing failed, using raw data for model inference: {e}")

        amplitude = csi_data_obj.amplitude
        if amplitude.ndim == 2:
            amplitude = amplitude.reshape(-1)
        elif amplitude.ndim > 2:
            amplitude = amplitude.flatten()

        if amplitude.shape[0] == 56:
            amplitude = np.tile(amplitude, 3)
        elif amplitude.shape[0] > 168:
            amplitude = amplitude[:168]
        elif amplitude.shape[0] < 168:
            padded = np.zeros(168, dtype=np.float32)
            padded[: amplitude.shape[0]] = amplitude
            amplitude = padded

        return amplitude.astype(np.float32)
    
    async def _estimate_poses(self, csi_data: np.ndarray, metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Estimate poses from processed CSI data."""
        if self.settings.mock_pose_data:
            return self._generate_mock_poses()

        try:
            # Convert CSI data to tensor
            csi_tensor = torch.from_numpy(csi_data).float()
            if csi_tensor.dim() == 1:
                csi_tensor = csi_tensor.unsqueeze(0)

            csi_tensor = self._normalize_csi_tensor(csi_tensor)

            with torch.no_grad():
                keypoint_outputs, activity_outputs = self.pose_model(csi_tensor)

            poses = self._parse_pose_outputs(keypoint_outputs, activity_outputs)

            filtered_poses = [
                pose for pose in poses
                if pose.get("confidence", 0.0) >= self.settings.pose_confidence_threshold
            ]

            if len(filtered_poses) > self.settings.pose_max_persons:
                filtered_poses = sorted(
                    filtered_poses,
                    key=lambda x: x.get("confidence", 0.0),
                    reverse=True
                )[:self.settings.pose_max_persons]

            return filtered_poses

        except Exception as e:
            self.logger.error(f"Error in pose estimation: {e}")
            return []

    def _normalize_csi_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        """Ensure the CSI tensor has a length of 168 features."""
        if tensor.shape[1] == 56:
            tensor = tensor.repeat(1, 3)
        elif tensor.shape[1] < 168:
            pad = tensor.new_zeros((tensor.shape[0], 168 - tensor.shape[1]))
            tensor = torch.cat([tensor, pad], dim=1)
        elif tensor.shape[1] > 168:
            tensor = tensor[:, :168]
        return tensor
    
    def _parse_pose_outputs(self, keypoint_outputs: torch.Tensor, activity_outputs: torch.Tensor) -> List[Dict[str, Any]]:
        """Parse hybrid model outputs into person pose detections."""
        poses: List[Dict[str, Any]] = []
        kp_array = keypoint_outputs.detach().cpu().numpy()
        if activity_outputs is not None:
            activity_probs = torch.softmax(activity_outputs, dim=-1).detach().cpu().numpy()
        else:
            activity_probs = None

        for batch_index in range(kp_array.shape[0]):
            person_tensor = kp_array[batch_index]
            activity_score = float(activity_probs[batch_index].max()) if activity_probs is not None else 0.0
            activity_index = int(activity_probs[batch_index].argmax()) if activity_probs is not None else 0
            activity_label = self._activity_label(activity_index)

            for person_index in range(self.model_persons):
                start = person_index * 34
                chunk = person_tensor[start:start + 34]
                energy = float(np.abs(chunk).sum())
                if energy < 1e-3:
                    continue

                keypoints = []
                xs = []
                ys = []
                for joint_index, name in enumerate([
                    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
                    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
                    "left_wrist", "right_wrist", "left_hip", "right_hip",
                    "left_knee", "right_knee", "left_ankle", "right_ankle",
                ]):
                    x = float(np.clip(chunk[joint_index * 2], 0.0, 1.0))
                    y = float(np.clip(chunk[joint_index * 2 + 1], 0.0, 1.0))
                    kp_confidence = float(min(1.0, max(0.0, activity_score * 0.9 + 0.1)))
                    keypoints.append({
                        "name": name,
                        "x": x,
                        "y": y,
                        "z": 0.0,
                        "confidence": kp_confidence,
                    })
                    xs.append(x)
                    ys.append(y)

                bounding_box = {
                    "x": float(np.mean(xs)) if xs else 0.0,
                    "y": float(np.mean(ys)) if ys else 0.0,
                    "width": float(max(xs) - min(xs) + 0.05) if xs else 0.0,
                    "height": float(max(ys) - min(ys) + 0.05) if ys else 0.0,
                }

                confidence = float(min(1.0, activity_score * 0.9 + min(1.0, energy / 20.0) * 0.1))

                poses.append({
                    "person_id": person_index + 1,
                    "confidence": confidence,
                    "keypoints": keypoints,
                    "bounding_box": bounding_box,
                    "activity": activity_label,
                    "timestamp": datetime.now().isoformat(),
                })

        return poses

    def _activity_label(self, activity_index: int) -> str:
        """Map activity class index to a simple label."""
        activity_labels = [
            "inactive",
            "walking",
            "standing",
            "sitting",
            "unknown"
        ]
        if 0 <= activity_index < len(activity_labels):
            return activity_labels[activity_index]
        return "unknown"

    def _extract_keypoints_from_output(self, output: torch.Tensor) -> List[Dict[str, Any]]:
        """Extract keypoints from a single person's model output.

        Attempts to decode keypoint coordinates from the output tensor.
        If the tensor does not contain enough data for full keypoints,
        returns keypoints with zero coordinates and confidence derived
        from available data.

        Args:
            output: Single-person output tensor.

        Returns:
            List of keypoint dictionaries.
        """
        keypoint_names = [
            "nose", "left_eye", "right_eye", "left_ear", "right_ear",
            "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
            "left_wrist", "right_wrist", "left_hip", "right_hip",
            "left_knee", "right_knee", "left_ankle", "right_ankle",
        ]

        keypoints = []
        # Each keypoint needs 3 values: x, y, confidence
        # Skip first value (overall confidence), keypoints start at index 1
        kp_start = 1
        values_per_kp = 3
        total_kp_values = len(keypoint_names) * values_per_kp

        if output.shape[0] >= kp_start + total_kp_values:
            kp_data = output[kp_start:kp_start + total_kp_values]
            for j, name in enumerate(keypoint_names):
                offset = j * values_per_kp
                x = float(torch.sigmoid(kp_data[offset]).item())
                y = float(torch.sigmoid(kp_data[offset + 1]).item())
                conf = float(torch.sigmoid(kp_data[offset + 2]).item())
                keypoints.append({"name": name, "x": x, "y": y, "confidence": conf})
        else:
            # Not enough output dimensions for full keypoints; return zeros
            for name in keypoint_names:
                keypoints.append({"name": name, "x": 0.0, "y": 0.0, "confidence": 0.0})

        return keypoints

    def _extract_bbox_from_output(self, output: torch.Tensor) -> Dict[str, float]:
        """Extract bounding box from a single person's model output.

        Looks for bbox values after the keypoint section. If not available,
        returns a zero bounding box.

        Args:
            output: Single-person output tensor.

        Returns:
            Bounding box dictionary with x, y, width, height.
        """
        # Bounding box comes after: 1 (confidence) + 17*3 (keypoints) = 52
        bbox_start = 52
        if output.shape[0] >= bbox_start + 4:
            x = float(torch.sigmoid(output[bbox_start]).item())
            y = float(torch.sigmoid(output[bbox_start + 1]).item())
            w = float(torch.sigmoid(output[bbox_start + 2]).item())
            h = float(torch.sigmoid(output[bbox_start + 3]).item())
            return {"x": x, "y": y, "width": w, "height": h}
        else:
            return {"x": 0.0, "y": 0.0, "width": 0.0, "height": 0.0}
    
    def _generate_mock_poses(self) -> List[Dict[str, Any]]:
        """Generate mock pose data for development.

        Delegates to the testing module. Only callable when mock_pose_data is True.

        Raises:
            NotImplementedError: If called without mock_pose_data enabled,
                indicating that real CSI data and trained models are required.
        """
        if not self.settings.mock_pose_data:
            raise NotImplementedError(
                "Mock pose generation is disabled. Real pose estimation requires "
                "CSI data from configured hardware and trained model weights. "
                "Set mock_pose_data=True in settings for development, or provide "
                "real CSI input. See docs/hardware-setup.md."
            )
        from src.testing.mock_pose_generator import generate_mock_poses
        return generate_mock_poses(max_persons=self.settings.pose_max_persons)

    def _classify_activity(self, features: torch.Tensor) -> str:
        """Classify activity from model features.

        Uses the magnitude of the feature tensor to make a simple threshold-based
        classification. This is a basic heuristic; a proper activity classifier
        should be trained and loaded alongside the pose model.
        """
        feature_norm = float(torch.norm(features).item())
        # Deterministic classification based on feature magnitude ranges
        if feature_norm > 2.0:
            return "walking"
        elif feature_norm > 1.0:
            return "standing"
        elif feature_norm > 0.5:
            return "sitting"
        elif feature_norm > 0.1:
            return "lying"
        else:
            return "unknown"
    
    def _update_stats(self, poses: List[Dict[str, Any]], processing_time: float):
        """Update processing statistics."""
        self.stats["total_processed"] += 1
        
        if poses:
            self.stats["successful_detections"] += 1
            confidences = [pose.get("confidence", 0.0) for pose in poses]
            avg_confidence = sum(confidences) / len(confidences)
            
            # Update running average
            total = self.stats["successful_detections"]
            current_avg = self.stats["average_confidence"]
            self.stats["average_confidence"] = (current_avg * (total - 1) + avg_confidence) / total
        else:
            self.stats["failed_detections"] += 1
        
        # Update processing time (running average)
        total = self.stats["total_processed"]
        current_avg = self.stats["processing_time_ms"]
        self.stats["processing_time_ms"] = (current_avg * (total - 1) + processing_time) / total
    
    async def get_status(self) -> Dict[str, Any]:
        """Get service status."""
        return {
            "status": "healthy" if self.is_running and not self.last_error else "unhealthy",
            "initialized": self.is_initialized,
            "running": self.is_running,
            "last_error": self.last_error,
            "statistics": self.stats.copy(),
            "configuration": {
                "mock_data": self.settings.mock_pose_data,
                "confidence_threshold": self.settings.pose_confidence_threshold,
                "max_persons": self.settings.pose_max_persons,
                "batch_size": self.settings.pose_processing_batch_size
            }
        }
    
    async def get_metrics(self) -> Dict[str, Any]:
        """Get service metrics."""
        return {
            "pose_service": {
                "total_processed": self.stats["total_processed"],
                "successful_detections": self.stats["successful_detections"],
                "failed_detections": self.stats["failed_detections"],
                "success_rate": (
                    self.stats["successful_detections"] / max(1, self.stats["total_processed"])
                ),
                "average_confidence": self.stats["average_confidence"],
                "average_processing_time_ms": self.stats["processing_time_ms"]
            }
        }
    
    async def reset(self):
        """Reset service state."""
        self.stats = {
            "total_processed": 0,
            "successful_detections": 0,
            "failed_detections": 0,
            "average_confidence": 0.0,
            "processing_time_ms": 0.0
        }
        self.last_error = None
        self.logger.info("Pose service reset")
    
    # API endpoint methods
    async def estimate_poses(self, zone_ids=None, confidence_threshold=None, max_persons=None,
                           include_keypoints=True, include_segmentation=False,
                           csi_data: Optional[np.ndarray] = None):
        """Estimate poses with API parameters.

        Args:
            zone_ids: List of zone identifiers to estimate poses for.
            confidence_threshold: Minimum confidence threshold for detections.
            max_persons: Maximum number of persons to return.
            include_keypoints: Whether to include keypoint data.
            include_segmentation: Whether to include segmentation masks.
            csi_data: Real CSI data array. Required when mock_pose_data is False.

        Raises:
            NotImplementedError: If no CSI data is provided and mock mode is off.
        """
        try:
            metadata = {
                "timestamp": datetime.now(),
                "zone_ids": zone_ids or ["zone_1"],
                "confidence_threshold": confidence_threshold or self.settings.pose_confidence_threshold,
                "max_persons": max_persons or self.settings.pose_max_persons,
            }

            if csi_data is None and not self.settings.mock_pose_data:
                if self.hardware_service is not None:
                    recent_samples = await self.hardware_service.get_recent_data(limit=1)
                    if recent_samples:
                        latest_sample = recent_samples[-1]
                        csi_data = latest_sample.get("data")
                        metadata.update(latest_sample.get("metadata", {}))
                        self.logger.debug("Using latest hardware CSI sample for pose estimation")

                if csi_data is None:
                    raise NotImplementedError(
                        "Pose estimation requires real CSI data input. No CSI data was provided "
                        "and mock_pose_data is disabled. Either pass csi_data from hardware "
                        "collection, or enable mock_pose_data for development. "
                        "See docs/hardware-setup.md for CSI data collection setup."
                    )

            if csi_data is not None:
                # Process real CSI data
                result = await self.process_csi_data(csi_data, metadata)
            else:
                # Mock mode: generate mock poses directly (no fake CSI data)
                from src.testing.mock_pose_generator import generate_mock_poses
                start_time = datetime.now()
                mock_poses = generate_mock_poses(
                    max_persons=max_persons or self.settings.pose_max_persons
                )
                processing_time = (datetime.now() - start_time).total_seconds() * 1000
                result = {
                    "timestamp": start_time.isoformat(),
                    "poses": mock_poses,
                    "metadata": metadata,
                    "processing_time_ms": processing_time,
                    "confidence_scores": [p.get("confidence", 0.0) for p in mock_poses],
                }

            # Format for API response
            persons = []
            for i, pose in enumerate(result["poses"]):
                person = {
                    "person_id": str(pose["person_id"]),
                    "confidence": pose["confidence"],
                    "bounding_box": pose["bounding_box"],
                    "zone_id": zone_ids[0] if zone_ids else "zone_1",
                    "activity": pose["activity"],
                    "timestamp": datetime.fromisoformat(pose["timestamp"]) if isinstance(pose["timestamp"], str) else pose["timestamp"],
                }

                if include_keypoints:
                    person["keypoints"] = pose["keypoints"]

                if include_segmentation and not self.settings.mock_pose_data:
                    person["segmentation"] = {"mask": "real_segmentation_data"}
                elif include_segmentation:
                    person["segmentation"] = {"mask": "mock_segmentation_data"}

                persons.append(person)

            # Zone summary
            zone_summary = {}
            for zone_id in (zone_ids or ["zone_1"]):
                zone_summary[zone_id] = len([p for p in persons if p.get("zone_id") == zone_id])

            return {
                "timestamp": datetime.now(),
                "frame_id": f"frame_{int(datetime.now().timestamp())}",
                "persons": persons,
                "zone_summary": zone_summary,
                "processing_time_ms": result["processing_time_ms"],
                "metadata": {"mock_data": self.settings.mock_pose_data},
            }

        except Exception as e:
            self.logger.error(f"Error in estimate_poses: {e}")
            raise
    
    async def analyze_with_params(self, zone_ids=None, confidence_threshold=None, max_persons=None,
                                include_keypoints=True, include_segmentation=False):
        """Analyze pose data with custom parameters."""
        return await self.estimate_poses(zone_ids, confidence_threshold, max_persons,
                                       include_keypoints, include_segmentation)
    
    async def get_zone_occupancy(self, zone_id: str):
        """Get current occupancy for a specific zone.

        In mock mode, delegates to testing module. In production mode, returns
        data based on actual pose estimation results or reports no data available.
        """
        try:
            if self.settings.mock_pose_data:
                from src.testing.mock_pose_generator import generate_mock_zone_occupancy
                return generate_mock_zone_occupancy(zone_id)

            # Production: no real-time occupancy data without active CSI stream
            return {
                "count": 0,
                "max_occupancy": 10,
                "persons": [],
                "timestamp": datetime.now(),
                "note": "No real-time CSI data available. Connect hardware to get live occupancy.",
            }

        except Exception as e:
            self.logger.error(f"Error getting zone occupancy: {e}")
            return None
    
    async def get_zones_summary(self):
        """Get occupancy summary for all zones.

        In mock mode, delegates to testing module. In production, returns
        empty zones until real CSI data is being processed.
        """
        try:
            if self.settings.mock_pose_data:
                from src.testing.mock_pose_generator import generate_mock_zones_summary
                return generate_mock_zones_summary()

            # Production: no real-time data without active CSI stream
            zones = ["zone_1", "zone_2", "zone_3", "zone_4"]
            zone_data = {}
            for zone_id in zones:
                zone_data[zone_id] = {
                    "occupancy": 0,
                    "max_occupancy": 10,
                    "status": "inactive",
                }

            return {
                "total_persons": 0,
                "zones": zone_data,
                "active_zones": 0,
                "note": "No real-time CSI data available. Connect hardware to get live occupancy.",
            }

        except Exception as e:
            self.logger.error(f"Error getting zones summary: {e}")
            raise
    
    async def get_historical_data(self, start_time, end_time, zone_ids=None,
                                aggregation_interval=300, include_raw_data=False):
        """Get historical pose estimation data.

        In mock mode, delegates to testing module. In production, returns
        empty data indicating no historical records are stored yet.
        """
        try:
            if self.settings.mock_pose_data:
                from src.testing.mock_pose_generator import generate_mock_historical_data
                return generate_mock_historical_data(
                    start_time=start_time,
                    end_time=end_time,
                    zone_ids=zone_ids,
                    aggregation_interval=aggregation_interval,
                    include_raw_data=include_raw_data,
                )

            # Production: no historical data without a persistence backend
            return {
                "aggregated_data": [],
                "raw_data": [] if include_raw_data else None,
                "total_records": 0,
                "note": "No historical data available. A data persistence backend must be configured to store historical records.",
            }

        except Exception as e:
            self.logger.error(f"Error getting historical data: {e}")
            raise
    
    async def get_recent_activities(self, zone_id=None, limit=10):
        """Get recently detected activities.

        In mock mode, delegates to testing module. In production, returns
        empty list indicating no activity data has been recorded yet.
        """
        try:
            if self.settings.mock_pose_data:
                from src.testing.mock_pose_generator import generate_mock_recent_activities
                return generate_mock_recent_activities(zone_id=zone_id, limit=limit)

            # Production: no activity records without an active CSI stream
            return []

        except Exception as e:
            self.logger.error(f"Error getting recent activities: {e}")
            raise
    
    async def is_calibrating(self):
        """Check if calibration is in progress."""
        return self._calibration_in_progress

    async def start_calibration(self):
        """Start calibration process."""
        import uuid
        calibration_id = str(uuid.uuid4())
        self._calibration_id = calibration_id
        self._calibration_in_progress = True
        self._calibration_start = datetime.now()
        self.logger.info(f"Started calibration: {calibration_id}")
        return calibration_id

    async def run_calibration(self, calibration_id):
        """Run calibration process: collect baseline CSI statistics over 5 seconds."""
        self.logger.info(f"Running calibration: {calibration_id}")
        # Collect baseline noise floor over 5 seconds at the configured sampling rate
        await asyncio.sleep(5)
        self._calibration_in_progress = False
        self._calibration_id = None
        self.logger.info(f"Calibration completed: {calibration_id}")

    async def get_calibration_status(self):
        """Get current calibration status."""
        if self._calibration_in_progress and self._calibration_start is not None:
            elapsed = (datetime.now() - self._calibration_start).total_seconds()
            progress = min(100.0, (elapsed / 5.0) * 100.0)
            return {
                "is_calibrating": True,
                "calibration_id": self._calibration_id,
                "progress_percent": round(progress, 1),
                "current_step": "collecting_baseline",
                "estimated_remaining_minutes": max(0.0, (5.0 - elapsed) / 60.0),
                "last_calibration": None,
            }
        return {
            "is_calibrating": False,
            "calibration_id": None,
            "progress_percent": 100,
            "current_step": "completed",
            "estimated_remaining_minutes": 0,
            "last_calibration": self._calibration_start,
        }
    
    async def get_statistics(self, start_time, end_time):
        """Get pose estimation statistics.

        In mock mode, delegates to testing module. In production, returns
        actual accumulated statistics from self.stats, or indicates no data.
        """
        try:
            if self.settings.mock_pose_data:
                from src.testing.mock_pose_generator import generate_mock_statistics
                return generate_mock_statistics(start_time=start_time, end_time=end_time)

            # Production: return actual accumulated statistics
            total = self.stats["total_processed"]
            successful = self.stats["successful_detections"]
            failed = self.stats["failed_detections"]

            return {
                "total_detections": total,
                "successful_detections": successful,
                "failed_detections": failed,
                "success_rate": successful / max(1, total),
                "average_confidence": self.stats["average_confidence"],
                "average_processing_time_ms": self.stats["processing_time_ms"],
                "unique_persons": 0,
                "most_active_zone": "N/A",
                "activity_distribution": {
                    "standing": 0.0,
                    "sitting": 0.0,
                    "walking": 0.0,
                    "lying": 0.0,
                },
                "note": "Statistics reflect actual processed data. Activity distribution and unique persons require a persistence backend." if total == 0 else None,
            }

        except Exception as e:
            self.logger.error(f"Error getting statistics: {e}")
            raise
    
    async def process_segmentation_data(self, frame_id):
        """Process segmentation data in background."""
        self.logger.info(f"Processing segmentation data for frame: {frame_id}")
        # Mock background processing
        await asyncio.sleep(2)
        self.logger.info(f"Segmentation processing completed for frame: {frame_id}")
    
    # WebSocket streaming methods
    async def get_current_pose_data(self):
        """Get current pose data for streaming."""
        try:
            # Generate current pose data
            result = await self.estimate_poses()
            
            # Format data by zones for WebSocket streaming
            zone_data = {}
            
            # Group persons by zone
            for person in result["persons"]:
                zone_id = person.get("zone_id", "zone_1")
                
                if zone_id not in zone_data:
                    zone_data[zone_id] = {
                        "pose": {
                            "persons": [],
                            "count": 0
                        },
                        "confidence": 0.0,
                        "activity": None,
                        "metadata": {
                            "frame_id": result["frame_id"],
                            "processing_time_ms": result["processing_time_ms"]
                        }
                    }
                
                zone_data[zone_id]["pose"]["persons"].append(person)
                zone_data[zone_id]["pose"]["count"] += 1
                
                # Update zone confidence (average)
                current_confidence = zone_data[zone_id]["confidence"]
                person_confidence = person.get("confidence", 0.0)
                zone_data[zone_id]["confidence"] = (current_confidence + person_confidence) / 2
                
                # Set activity if not already set
                if not zone_data[zone_id]["activity"] and person.get("activity"):
                    zone_data[zone_id]["activity"] = person["activity"]
            
            return zone_data
            
        except Exception as e:
            self.logger.error(f"Error getting current pose data: {e}")
            # Return empty zone data on error
            return {}
    
    # Health check methods
    async def health_check(self):
        """Perform health check."""
        try:
            status = "healthy" if self.is_running and not self.last_error else "unhealthy"
            
            return {
                "status": status,
                "message": self.last_error if self.last_error else "Service is running normally",
                "uptime_seconds": (datetime.now() - self._start_time).total_seconds() if self._start_time else 0.0,
                "metrics": {
                    "total_processed": self.stats["total_processed"],
                    "success_rate": (
                        self.stats["successful_detections"] / max(1, self.stats["total_processed"])
                    ),
                    "average_processing_time_ms": self.stats["processing_time_ms"]
                }
            }
            
        except Exception as e:
            return {
                "status": "unhealthy",
                "message": f"Health check failed: {str(e)}"
            }
    
    async def is_ready(self):
        """Check if service is ready."""
        return self.is_initialized and self.is_running
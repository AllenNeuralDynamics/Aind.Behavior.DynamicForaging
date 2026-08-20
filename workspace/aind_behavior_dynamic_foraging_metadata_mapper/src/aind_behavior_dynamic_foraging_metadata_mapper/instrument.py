import logging
import os
from datetime import date
from decimal import Decimal
from pathlib import Path
from typing import Literal, Optional

from aind_behavior_dynamic_foraging.data_contract import dataset as df_foraging_dataset
from aind_behavior_dynamic_foraging.rig import AindDynamicForagingRig
from aind_behavior_services.rig import water_valve as abs_water_valve
from aind_behavior_services.utils import get_fields_of_type, utcnow
from aind_data_schema.components.connections import Connection
from aind_data_schema.components.coordinates import Axis, AxisName, CoordinateSystem, Direction, Origin, Scale
from aind_data_schema.components.devices import (
    AnatomicalRelative,
    Camera,
    CameraAssembly,
    CameraChroma,
    CameraTarget,
    Cooling,
    DataInterface,
    Device,
    Enclosure,
    HarpDevice,
    HarpDeviceType,
    Lens,
    LightEmittingDiode,
    LickSensorType,
    LickSpout,
    LickSpoutAssembly,
    MotorizedStage,
    SizeUnit,
    Tube,
)
from aind_data_schema.components.measurements import CalibrationFit, FitType, GenericModel, VolumeCalibration
from aind_data_schema.core.acquisition import CALIBRATIONS
from aind_data_schema.core.instrument import Instrument
from aind_data_schema_models.modalities import Modality
from aind_data_schema_models.organizations import Organization
from aind_data_schema_models.units import FrequencyUnit, TimeUnit, VolumeUnit
from clabe.data_mapper.aind_data_schema import AindDataSchemaRigDataMapper
from contraqctor.contract import DataStream

logger = logging.getLogger(__name__)


class AindInstrumentDataMapper(AindDataSchemaRigDataMapper):
    def __init__(
        self,
        data_path: os.PathLike,
    ):
        """
        Create Instrument model for completed session.

        Args:
            data_directory (os.PathLike):
                Path to the directory containing the dataset to analyze. This
                directory is expected to include all required behavioral data files.
        """

        super().__init__()
        self._data_path = Path(data_path)

    @staticmethod
    def _get_water_calibration(rig_model: AindDynamicForagingRig) -> list[CALIBRATIONS]:

        water_calibrations = get_fields_of_type(rig_model, abs_water_valve.WaterValveCalibration)
        vol_cal = []
        for device_name, wc in water_calibrations:
            if device_name and wc.interval_average:
                vol_cal.append(
                    VolumeCalibration(
                        device_name=device_name,
                        calibration_date=wc.date if wc.date else utcnow(),
                        input=list(wc.interval_average.keys()),
                        output=list(wc.interval_average.values()),
                        input_unit=TimeUnit.S,
                        output_unit=VolumeUnit.ML,
                        fit=CalibrationFit(
                            fit_type=FitType.LINEAR,
                            fit_parameters=GenericModel.model_validate(wc.model_dump()),
                        ),
                    )
                )
        return vol_cal

    def rig_schema(self):
        return self.mapped

    @staticmethod
    def compose_version(
        harp_stream: DataStream, version_prefix: Literal["Core", "Firmware", "Hardware"]
    ) -> Optional[str]:
        """
        Compose versioning from high and low version

        Args:
            harp_stream (DataStream):
               Data stream for device
            version_prefix (str):
               Stream name prefix
        """
        try:
            high = harp_stream[f"{version_prefix}VersionHigh"].data.iloc[0, 0]
            low = harp_stream[f"{version_prefix}VersionLow"].data.iloc[0, 0]
            return f"{high}.{low}"

        except FileNotFoundError as e:
            logger.warning(f"No {version_prefix} version found for {harp_stream.name}: {str(e)}")
            return

    @property
    def session_name(self):
        raise NotImplementedError("Method not implemented.")

    def map(self) -> Instrument:
        logger.info("Mapping aind-data-schema Instrument.")
        self._mapped = self._map()
        return self.mapped

    def _map(self) -> Instrument:
        """
        Create Instrument model for completed session.

        Returns:
            Instrument:
                Instrument model for session

        Raises:
            FileNotFoundError:
                If the specified data directory or required files do not exist.

            ValueError:
                If the dataset is malformed or missing required fields for
                computing metrics.
        """

        dataset = df_foraging_dataset(self._data_path)
        input_schemas = dataset["Behavior"]["InputSchemas"]
        rig = AindDynamicForagingRig.model_validate(input_schemas["Rig"].data)

        components = []
        connections = []

        # cameras
        controller = rig.triggered_camera_controller
        fps = float(controller.frame_rate) if controller.frame_rate else float("nan")
        for name, cam in rig.triggered_camera_controller.cameras.items():
            camera = Camera(
                name=name,
                manufacturer=Organization.FLIR,
                chroma=CameraChroma.BW,
                cooling=Cooling.NO_COOLING,
                data_interface=DataInterface.USB,
                sensor_format="1/2.9",
                sensor_format_unit=SizeUnit.IN,
                sensor_width=720,
                sensor_height=540,
                model="Blackfly S BFS-U3-04S2M",
                frame_rate=Decimal(str(fps)),
                frame_rate_unit=FrequencyUnit.HZ,
                gain=Decimal(str(cam.gain) if cam.gain is not None else "0"),
                serial_number=cam.serial_number,
                crop_offset_x=cam.region_of_interest.x if cam.region_of_interest.x > 0 else None,
                crop_offset_y=cam.region_of_interest.y if cam.region_of_interest.y > 0 else None,
                crop_width=cam.region_of_interest.width if cam.region_of_interest.width > 0 else None,
                crop_height=cam.region_of_interest.height if cam.region_of_interest.height > 0 else None,
                crop_unit=SizeUnit.PX,
                additional_settings=GenericModel.model_validate(cam.model_dump()),
            )
            assembly = CameraAssembly(
                name=f"{name}Assembly",
                camera=camera,
                target=CameraTarget.BODY if "Body" in name else CameraTarget.FACE,
                lens=Lens(name="Lens A", manufacturer=Organization.FUJINON),
                relative_position=[AnatomicalRelative.RIGHT if "Body" in name else AnatomicalRelative.SUPERIOR],
            )
            components.append(assembly)

        # behavior board
        behavior_board = dataset["Behavior"]["HarpBehavior"].load()
        components.append(
            HarpDevice(
                name="BehaviorBoard",
                harp_device_type=HarpDeviceType.BEHAVIOR,
                serial_number=rig.harp_behavior.serial_number,
                manufacturer=Organization.CHAMPALIMAUD,
                is_clock_generator=False,
                firmware_version=self.compose_version(behavior_board, "Firmware"),
                hardware_version=self.compose_version(behavior_board, "Hardware"),
                core_version=self.compose_version(behavior_board, "Core"),
            )
        )

        # clock generator
        clock_generator = dataset["Behavior"]["HarpClockGenerator"].load()
        components.append(
            HarpDevice(
                name="ClockGenerator",
                harp_device_type=HarpDeviceType.WHITERABBIT,
                serial_number=rig.harp_clock_generator.serial_number,
                is_clock_generator=True,
                firmware_version=self.compose_version(clock_generator, "Firmware"),
                hardware_version=self.compose_version(clock_generator, "Hardware"),
                core_version=self.compose_version(clock_generator, "Core"),
            )
        )

        # sound card
        sound_card = dataset["Behavior"]["HarpSoundCard"].load()
        components.append(
            HarpDevice(
                name="SoundCard",
                harp_device_type=HarpDeviceType.SOUNDCARD,
                serial_number=rig.harp_sound_card.serial_number,
                manufacturer=Organization.CHAMPALIMAUD,
                is_clock_generator=False,
                firmware_version=self.compose_version(sound_card, "Firmware"),
                hardware_version=self.compose_version(sound_card, "Hardware"),
                core_version=self.compose_version(sound_card, "Core"),
            )
        )

        # optional harp devices
        if rig.harp_lickometer_left:
            left = dataset["Behavior"]["HarpLickometerLeft"].load()
            components.append(
                HarpDevice(
                    name="LickometerLeft",
                    harp_device_type=HarpDeviceType.LICKETYSPLIT,
                    serial_number=rig.harp_lickometer_left.serial_number,
                    is_clock_generator=False,
                    firmware_version=self.compose_version(left, "Firmware"),
                    hardware_version=self.compose_version(left, "Hardware"),
                    core_version=self.compose_version(left, "Core"),
                )
            )
        if rig.harp_lickometer_right:
            right = dataset["Behavior"]["HarpLickometerRight"].load()
            components.append(
                HarpDevice(
                    name="LickometerRight",
                    serial_number=rig.harp_lickometer_right.serial_number,
                    harp_device_type=HarpDeviceType.LICKETYSPLIT,
                    is_clock_generator=False,
                    firmware_version=self.compose_version(right, "Firmware"),
                    hardware_version=self.compose_version(right, "Hardware"),
                    core_version=self.compose_version(right, "Core"),
                )
            )
        if rig.harp_sniff_detector:
            sniff = dataset["Behavior"]["HarpSniffDetector"].load()
            components.append(
                HarpDevice(
                    name="SniffDetector",
                    harp_device_type=HarpDeviceType.SNIFFDETECTOR,
                    serial_number=rig.harp_sniff_detector.serial_number,
                    is_clock_generator=False,
                    firmware_version=self.compose_version(sniff, "Firmware"),
                    hardware_version=self.compose_version(sniff, "Hardware"),
                    core_version=self.compose_version(sniff, "Core"),
                )
            )
        if rig.harp_environment_sensor:
            env_sen = dataset["Behavior"]["HarpEnvironmentSensor"].load()
            components.append(
                HarpDevice(
                    name="EnvironmentSensor",
                    harp_device_type=HarpDeviceType.ENVIRONMENTSENSOR,
                    serial_number=rig.harp_environment_sensor.serial_number,
                    is_clock_generator=False,
                    firmware_version=self.compose_version(env_sen, "Firmware"),
                    hardware_version=self.compose_version(env_sen, "Hardware"),
                    core_version=self.compose_version(env_sen, "Core"),
                )
            )

        # lick spout assembly
        components.append(
            LickSpoutAssembly(
                name="LickSpoutAssembly",
                lick_spouts=[
                    LickSpout(
                        name="Left lick spout",
                        manufacturer=Organization.OTHER,
                        spout_diameter=Decimal("1.2"),
                        spout_diameter_unit=SizeUnit.MM,
                        solenoid_valve=Device(
                            name="Solenoid Left",
                            manufacturer=Organization.THE_LEE_COMPANY,
                            model="LHDB1233518H",
                        ),
                        lick_sensor=Device(
                            name="Lick Sensor Left",
                            manufacturer=Organization.JANELIA,
                        ),
                        lick_sensor_type=LickSensorType.CAPACITIVE,
                    ),
                    LickSpout(
                        name="Right lick spout",
                        manufacturer=Organization.OTHER,
                        spout_diameter=Decimal("1.2"),
                        spout_diameter_unit=SizeUnit.MM,
                        solenoid_valve=Device(
                            name="Solenoid Right",
                            manufacturer=Organization.THE_LEE_COMPANY,
                            model="LHDB1233518H",
                        ),
                        lick_sensor=Device(
                            name="Lick Sensor Right",
                            manufacturer=Organization.JANELIA,
                        ),
                        lick_sensor_type=LickSensorType.CAPACITIVE,
                    ),
                ],
                motorized_stage=MotorizedStage(
                    name="motorized_stage",
                    manufacturer=Organization.AIND,
                    model="328-300-00",
                    travel=Decimal("30"),
                    travel_unit=SizeUnit.MM,
                    notes="This stage is driven by the manipulator device.",
                ),
            )
        )

        # mouse platform
        components.append(
            Tube(
                name="mouse_tube_foraging",
                manufacturer=Organization.CUSTOM,
                diameter=Decimal("3.0"),
                diameter_unit=SizeUnit.CM,
            )
        )

        # IR light source
        components.append(
            LightEmittingDiode(
                name="IR LED",
                manufacturer=Organization.THORLABS,
                model="M810L5",
                wavelength=810,
            )
        )

        # enclosure
        components.append(
            Enclosure(
                name="Behavior enclosure",
                manufacturer=Organization.AIND,
                size=Scale(scale=[54, 54, 54]),
                size_unit=SizeUnit.CM,
                internal_material="",
                external_material="",
                grounded=False,
                laser_interlock=False,
                air_filtration=False,
                notes=" (v1v2 upgrade): Scale is width/length/height",
            )
        )

        components.append(
            Device(
                name="speaker",
                manufacturer=Organization.TYMPHANY,
                model="XT25SC90-04",
            )
        )

        # connections
        for name in rig.triggered_camera_controller.cameras:
            connections.append(
                Connection(
                    source_device="BehaviorBoard",
                    target_device=name,
                )
            )
        connections.append(
            Connection(
                source_device="BehaviorBoard",
                target_device="Solenoid Left",
                source_port="SupplyPort0",
            )
        )
        connections.append(
            Connection(source_device="BehaviorBoard", target_device="Solenoid Right", source_port="SupplyPort1")
        )

        return Instrument(
            instrument_id=rig.rig_name,
            modification_date=date.today(),
            modalities=[Modality.BEHAVIOR, Modality.BEHAVIOR_VIDEOS],
            coordinate_system=CoordinateSystem(
                name="RigCoordinateSystem",
                origin=Origin.ORIGIN,
                axes=[
                    Axis(name=AxisName.X, direction=Direction.LR),
                    Axis(name=AxisName.Y, direction=Direction.FB),
                    Axis(name=AxisName.Z, direction=Direction.DU),
                ],
                axis_unit=SizeUnit.MM,
            ),
            components=components,
            connections=connections,
            calibrations=self._get_water_calibration(rig),
        )

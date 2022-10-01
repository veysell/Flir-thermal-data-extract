import numpy as np
from nptyping import Array
from math import sqrt, exp, fsum
from typing import Optional, Dict, Union

class FlirThermogram:
    # Required members variables required to be set
    __thermal: Array[np.int64, ..., ...]
    __metadata_adjustments: Dict[str, Union[float, int]]
    path: Optional[str]

    def __init__(
        self,
        thermal: Array[np.int64, ..., ...],
        metadata: Dict[str, Union[float, int]],
        metadata_adjustments: Dict[str, Union[float, int]] = {},
    ):
        self.__thermal = thermal  # Raw thermal data
        self.__metadata = metadata.copy()
        self.__metadata_adjustments = metadata_adjustments.copy()
        

    @property
    def kelvin(self) -> Array[np.float64, ..., ...]:
        return self.__raw_to_kelvin_with_metadata(self.__thermal)

    @property
    def celsius(self) -> Array[np.float64, ..., ...]:
        return self.kelvin - 273.15

    @property
    def fahrenheit(self) -> Array[np.float64, ..., ...]:
        return self.celsius * 1.8 + 32.00

    def __raw_to_kelvin_with_metadata(
        self, thermal, orig: bool = False
    ) -> Array[np.float64, ..., ...]:
        metadata = self.__metadata.copy()
        if not orig:
            metadata.update(self.__metadata_adjustments)

        return FlirThermogram.__raw_to_kelvin(
            thermal,
            metadata["emissivity"],
            metadata["object_distance"],
            metadata["atmospheric_temperature"],
            metadata["ir_window_temperature"],
            metadata["ir_window_transmission"],
            metadata["reflected_apparent_temperature"],
            metadata["relative_humidity"],
            metadata["planck_r1"],
            metadata["planck_r2"],
            metadata["planck_b"],
            metadata["planck_f"],
            metadata["planck_o"],
            metadata["atmospheric_trans_alpha1"],
            metadata["atmospheric_trans_alpha2"],
            metadata["atmospheric_trans_beta1"],
            metadata["atmospheric_trans_beta2"],
            metadata["atmospheric_trans_x"],
        )

    @staticmethod
    def __raw_to_kelvin(
        thermal,
        emissivity,
        object_distance,
        atmospheric_temperature,
        ir_window_temperature,
        ir_window_transmission,
        reflected_apparent_temperature,
        relative_humidity,
        planck_r1,
        planck_r2,
        planck_b,
        planck_f,
        planck_o,
        atmospheric_trans_alpha1,
        atmospheric_trans_alpha2,
        atmospheric_trans_beta1,
        atmospheric_trans_beta2,
        atmospheric_trans_x,
    ) -> Array[np.float64, ..., ...]:
        # Transmission through window (calibrated)
        emiss_wind = 1 - ir_window_transmission
        refl_wind = 0

        # Transmission through the air
        water = relative_humidity * exp(
            1.5587
            + 0.06939 * (atmospheric_temperature - 273.15)
            - 0.00027816 * (atmospheric_temperature - 273.17) ** 2
            + 0.00000068455 * (atmospheric_temperature - 273.15) ** 3
        )

        def calc_atmos(alpha, beta):
            term1 = -sqrt(object_distance / 2)
            term2 = alpha + beta * sqrt(water)
            return exp(term1 * term2)

        atmos1 = calc_atmos(atmospheric_trans_alpha1, atmospheric_trans_beta1)
        atmos2 = calc_atmos(atmospheric_trans_alpha2, atmospheric_trans_beta2)
        tau1 = atmospheric_trans_x * atmos1 + (1 - atmospheric_trans_x) * atmos2
        tau2 = atmospheric_trans_x * atmos1 + (1 - atmospheric_trans_x) * atmos2

        # Radiance from the environment
        def plancked(t):
            planck_tmp = planck_r2 * (exp(planck_b / t) - planck_f)
            return planck_r1 / planck_tmp - planck_o

        raw_refl1 = plancked(reflected_apparent_temperature)
        raw_refl1_attn = (1 - emissivity) / emissivity * raw_refl1

        raw_atm1 = plancked(atmospheric_temperature)
        raw_atm1_attn = (1 - tau1) / emissivity / tau1 * raw_atm1

        term3 = emissivity * tau1 * ir_window_transmission
        raw_wind = plancked(ir_window_temperature)
        raw_wind_attn = emiss_wind / term3 * raw_wind

        raw_refl2 = plancked(reflected_apparent_temperature)
        raw_refl2_attn = refl_wind / term3 * raw_refl2

        raw_atm2 = plancked(atmospheric_temperature)
        raw_atm2_attn = (1 - tau2) / term3 / tau2 * raw_atm2

        subtraction = fsum(
            [raw_atm1_attn, raw_atm2_attn, raw_wind_attn, raw_refl1_attn, raw_refl2_attn]
        )

        raw_obj = thermal.astype(np.float64)
        raw_obj /= emissivity * tau1 * ir_window_transmission * tau2
        raw_obj -= subtraction

        # Temperature from radiance
        raw_obj += planck_o
        raw_obj *= planck_r2
        planck_term = planck_r1 / raw_obj + planck_f
        return planck_b / np.log(planck_term)

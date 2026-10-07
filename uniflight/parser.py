import json
import numpy as np
import pandas as pd
import utils.containers

class Parser:
    def __init__(self, rocket):
        json_filename = "rockets/" + rocket + "/constant.json"

        with open(json_filename, "r") as r_file:
            r_data = json.load(r_file)

        # common data
        self.name              = r_data["name"]
        self.payload_mass      = r_data["payload_mass"]
        self.block_mass        = r_data["block_mass"]
        self.exhaust_velocity  = r_data["exhaust_velocity"]
        self.thrust            = r_data["thrust"]
        self.structural_values = r_data["structural_values"]
        self.fuel_type         = r_data["fuel_type"]
        self.oxidizer_type     = r_data["oxidizer_type"]
        self.boosters_number   = r_data["booster_number"]
        self.attack_coefs      = r_data["attack_coefs"]
        self.chumbers_number   = r_data["chumbers_number"]
        self.prop_reserve      = r_data["prop_reserve"]
        self.coast_phase       = r_data["coast_phase"]

        is_equal = len(set(map(len,
            [self.oxidizer_type,
             self.fuel_type,
             self.structural_values,
             self.thrust,
             self.exhaust_velocity,
             self.block_mass,
             self.prop_reserve,
             self.coast_phase]))) == 1

        if not is_equal:
            raise ValueError("Не хватает данных для всех ступеней!")

        self.csv_filename = "rockets/" + rocket + "/distributed.csv"

        self.distributed = utils.containers.LengthData()

        df = pd.read_csv(self.csv_filename)
        self.distributed.lengths    = pd.to_numeric(df["L"], errors='coerce').tolist()
        self.distributed.diameters  = pd.to_numeric(df["D"], errors='coerce').tolist()
        self.distributed.classes    = df["Class"].tolist()
        self.distributed.stages     = df["Stage"].tolist()
        self.distributed.cumlengths = np.cumsum(self.distributed.lengths).tolist()
        self.distributed.numbers    = list(range(len(self.distributed.lengths)))

        if len(self.distributed.cumlengths) != len(set(self.distributed.cumlengths)):
            raise ValueError("Найдены одинаковые кумулятивные длины")
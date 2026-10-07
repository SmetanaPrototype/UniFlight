class LengthData:
    def __init__(self):
        self.numbers     = []
        self.lengths     = []
        self.cumlengths  = []
        self.diameters   = []
        self.classes     = []
        self.stages      = []
        self.stiffnesses = []
        self.areas       = []
        self.volumes     = []
        self.masses      = []
        self.coords      = []


class TimeData:
    def __init__(self):
        self.thrusts      = []
        self.masses       = []
        self.distromasses = []
        self.times        = []
        self.statics      = []
        self.inertionsX   = []
        self.inertionsY   = []
        self.centers      = []


class Block:
    def __init__(self):
        self.time_data   = TimeData()
        self.length_data = LengthData()
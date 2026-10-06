available_rockets =  ['soyuz21v', 'soyuz21b', 'amur', 'juno2']

mode_num = 3

timestep = 2
lenstep  = 0.01

tail_coefficient = 1/3
accuracy = 1e-9

def get_classes_list():
    return ["Tail", "Fuel", "Oxidizer", "Construction", "Head"]

def get_stages_list(block_num):
    if block_num not in [2,3,4,5]:
        raise ValueError("Current block number is not supported")

    if block_num == 2:
        return ["First", "Second", "Payload"]
    elif block_num == 3:
        return ["First", "Second", "Third", "Payload"]
    elif block_num == 4:
        return ["First", "Second", "Third", "Fourth", "Payload"]
    elif block_num == 5:
        return ["First", "Second", "Third", "Fourth", "Fifth", "Payload"]
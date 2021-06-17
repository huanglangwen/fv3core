class Microphysics:
    def __init__(self, state):
        self._inputs = ["gq0", "gt0"]

    def __call__(self, state):
        pass
        # gfdl_cloud_microphys_driver(state[self._inputs[0]])

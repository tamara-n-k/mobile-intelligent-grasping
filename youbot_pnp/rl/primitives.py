class StubPrimitive:
    def execute(self, target):
        return True, 0.0   # (success, sim_time)


'''
The parameter policy (your low-level policy) outputs an approximate x, y that selects a cube, and grasp then replaces it with the cube's pose from the simulator state.

1. Task policy: chooses to run grasp.
2. Parameter policy: outputs a target, which only has to land within 2 cm of a cube.
3. grasp: finds the nearest cube that isn't placed yet and checks the 2 cm limit.
4. Simulator state: provides the exact position and yaw used for the IK, planning and motion.
'''
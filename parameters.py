import numpy as np

class Parameters:

    def __init__(self, map_contour):
        self.map_contour = map_contour
        self.use_workspace_validator = True

        self.scale = 0.01 # cm to m
        # Angular bounds in radians w.r.t camera
        self.theta_x = 180.0 * (np.pi/180)
        self.theta_y = 180.0 * (np.pi/180)
        self.theta_z = 180.0 * (np.pi/180)

        # Position bound in meters
        self.pbound_x = 40.0
        self.pbound_y = 40.0
        self.pbound_z = 40.0

        # Threshold for maximum position value for generating random position
        self.max_posValue = 30.0 # meters

        # place radius
        self.place_radius = 3.0 # meters
        self.use_SE2 = False
        self.place_angularThreshold = np.array([np.deg2rad(10),
                                                np.deg2rad(60),
                                                np.deg2rad(10)])

        # Camera intrinsics
        self.intrinsics = [699.778, 699.778, 596.412, 371.069, 0.120008]
        # Database intrinsics
        self.db_intrinsics = [1399.56, 1399.56, 875.825, 565.139, 0.120008]

        # intrinsic matrices: observation and dB
        self.K_mat = np.array([[self.intrinsics[0], 0., self.intrinsics[2]],
                               [0., self.intrinsics[1], self.intrinsics[3]],
                               [0., 0., 1.]])

        self.dBK_mat = np.array([[self.db_intrinsics[0], 0., self.db_intrinsics[2]],
                                 [0., self.db_intrinsics[1], self.db_intrinsics[3]],
                                 [0., 0., 1.]])

        # Use semantic SLAM pose as initial solution to set bound
        self.use_initSol = False
        # plot output positions and path
        self.plot_output = True

        # Optimization parameters
        self.max_iterations = 100
        self.pop_size = 100
        self.dim = 6
        self.runs = 1

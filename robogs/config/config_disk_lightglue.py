CONFIG = {
    # PyGLOMAP pipeline configs
    'VIEW_GRAPH_CALIBRATOR_OPTIONS': {
        'thres_lower_ratio': 0.1,
        'thres_higher_ratio': 10,
        'thres_two_view_error': 2.,
        'thres_loss_function': 1e-2,
        'max_num_iterations': 100,
        'function_tolerance': 1e-5,
    },
    'RANSAC_OPTIONS': {
        'max_iterations': 50000
    },
    'BUNDLE_OPTIONS': {},
    'INLIER_THRESHOLD_OPTIONS': {
        'max_angle_error': 1.,
        'max_reprojection_error': 1e-1,
        'min_triangulation_angle': 1.,
        'max_epipolar_error_E': 1.,
        'max_epipolar_error_F': 4.,
        'max_epipolar_error_H': 4.,
        'min_inlier_num': 50,
        'min_inlier_ratio': 0.5,
        'max_rotation_error': 5.,
    },
    'ROTATION_ESTIMATOR_OPTIONS': {
        'max_num_l1_iterations': 5,
        'l1_step_convergence_threshold': 0.001,
        'max_num_irls_iterations': 100,
        'irls_step_convergence_threshold': 0.001,
        'axis': [0, 1, 0],
        'irls_loss_parameter_sigma': 5.0,
        'weight_type': 'GEMAN_MCCLURE',
        'use_weight': False,
        'verbose': False,
    },
    'L1_SOLVER_OPTIONS': {
        'max_num_iterations': 1000,
        'rho': 1.0,
        'alpha': 1.0,
        'absolute_tolerance': 1e-4,
        'relative_tolerance': 1e-2,
    },
    'TRACK_ESTABLISHMENT_OPTIONS': {
        'thres_inconsistency': 10.,
        'min_num_tracks_per_view': 2**32-1,
        'min_num_view_per_track': 3,
        'max_num_view_per_track': 100,
        'max_num_tracks': 10000000,
    },
    'GLOBAL_POSITIONER_OPTIONS': {
        'min_num_view_per_track': 3,
        'seed': 1,
        'constraint_type': 'ONLY_POINTS',
        'constraint_reweight_scale': 1.0,
        'thres_loss_function': 1e-1,
        'max_num_iterations': 100,
        'function_tolerance': 1e-5,
    },
    'BUNDLE_ADJUSTER_OPTIONS': {
        'optimize_rotations': True,
        'optimize_translation': True,
        'optimize_intrinsics': True,
        'optimize_points': True,
        'min_num_view_per_track': 3,
        'thres_loss_function': 1.,
        'max_num_iterations': 200,
        'function_tolerance': 1e-5,
    },
    # feature handler configs
    'FEATURE_HANDLER_OPTIONS': {
        'min_num_matches': 60,
    },
}

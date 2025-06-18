
import torch

def create_transformation_matrix_mdh(theta, a, alpha, d):
    """
    Generate individual transformation matrix using MDH parameters.
    """
    return torch.tensor([
        [torch.cos(theta), -torch.sin(theta), 0, a],
        [torch.sin(theta)*torch.cos(alpha), torch.cos(theta)*torch.cos(alpha), -torch.sin(alpha), -d*torch.sin(alpha)],
        [torch.sin(theta)*torch.sin(alpha), torch.cos(theta)*torch.sin(alpha), torch.cos(alpha), d*torch.cos(alpha)],
        [0, 0, 0, 1]
    ])

def reflect_axis(T,reflection_matrix=None):
    """
    Reflect the transformation matrix from y up to z up; supersplat to mujoco
    """

    if reflection_matrix is None:
        reflection_matrix = torch.tensor([[1, 0, 0, 0],
                                       [0, 0, 1, 0],
                                       [0, -1, 0, 0],
                                       [0, 0, 0, 1]]).type_as(T)
    else: 
        reflection_matrix=reflection_matrix
    return reflection_matrix@T@reflection_matrix

def calculate_franka_mdh_pre_frame(thetas):
    """
    Generate the deformation of Franka Emika Panda MDH parameters for pre-frame kinematics.
    Theta is set to zero for initial values; a, alpha, and d parameters from the original documentation.
    """

    # Franka Emika Panda MDH parameters (from documentation)
    # Initial theta (radians) set to zero
    thetas = thetas



    # a = torch.tensor([0.06, 0.0, 0.05, 0.155, -0.0825, 0, 0.088-0.05
    #                   , 0.102,
    #                   -0.0182,0,0,0
    #                   ])
    # alpha = torch.tensor([0, -torch.pi/2, -torch.pi/2+0.05, torch.pi/2, -0.11-torch.pi/2, torch.pi/2, torch.pi/2,
    #                       torch.pi,
    #                       -torch.pi/2,-torch.pi/2,-torch.pi/2,-torch.pi/2
    #                       ])
    # d = torch.tensor([0.46, -0.07, -0.48, -0.08, -0.584, -0.1, 0, 
    #                   0.112,
    #                   -0.014,0.014,-0.014,-0.014
    #                 ])
    

    #n
    a = torch.tensor([0.12, -0.05, -0.05, 0.125, -0.2125, -0.2, 0.23
                      , 0,
                      0,0,0,0,0
                      ])
    alpha = torch.tensor([0, -torch.pi/2, -torch.pi/2, torch.pi/2, -torch.pi/2, torch.pi/2, torch.pi/2,
                          0,
                          0,0,0,0,0
                          ])
    #n-1
    d = torch.tensor([0.69, -0.08, -0.66, 0.1, -0.744, 0.14, 0, 
                      0.0,
                      0.0,0.0,0.0,0.0,0
                    ])


    transformations = []
    # Store the cumulative transformations from base up to each joint
    final_transformations_list = []

    # First handle joints 0 through 7 (arm)
    final_transformation = torch.eye(4)
    


    # for the arm part, which is from 0 to 7, follow this way

    for i in range(12):
        T_temp = create_transformation_matrix_mdh(thetas[i], a[i], alpha[i], d[i])
        T_temp = reflect_axis(T_temp)
        transformations.append(T_temp)  # store the raw MDH for each joint
        # Multiply cumulatively
        final_transformation = torch.mm(final_transformation, T_temp)
        final_transformations_list.append(final_transformation.clone())

    # Save the transform at joint 7
    T7 = final_transformation.clone()

    # Now handle the finger joints 8 through 11


    # the logic of gripper


    # for i in range(9, 12):
    #     T_temp = create_transformation_matrix_mdh(thetas[i], a[i], alpha[i], d[i])
    #     T_temp = reflect_axis(T_temp)
    #     transformations.append(T_temp)  
    #     # Each finger frame i is computed from T7, 
    #     # not from the previous finger frame
    #     T_finger = torch.mm(T7, T_temp)
    #     final_transformations_list.append(T_finger.clone())

    return transformations, final_transformations_list


def calculate_franka_with_gripper_mdh_pre_frame(thetas,gripper_type='robotiq2f-85 '):
    """
    Generate the deformation of Franka Emika Panda MDH parameters for pre-frame kinematics.
    Theta is set to zero for initial values; a, alpha, and d parameters from the original documentation.
    """

    # Franka Emika Panda MDH parameters (from documentation)
    # Initial theta (radians) set to zero
    thetas = thetas



    # a = torch.tensor([0.06, 0.0, 0.05, 0.155, -0.0825, 0, 0.088-0.05
    #                   , 0.102,
    #                   -0.0182,0,0,0
    #                   ])
    # alpha = torch.tensor([0, -torch.pi/2, -torch.pi/2+0.05, torch.pi/2, -0.11-torch.pi/2, torch.pi/2, torch.pi/2,
    #                       torch.pi,
    #                       -torch.pi/2,-torch.pi/2,-torch.pi/2,-torch.pi/2
    #                       ])
    # d = torch.tensor([0.46, -0.07, -0.48, -0.08, -0.584, -0.1, 0, 
    #                   0.112,
    #                   -0.014,0.014,-0.014,-0.014
    #                 ])
    
    if add_gripper:
            joint_angles_degrees_gripper = np.zeros(5)
            a_gripper = np.zeros(5)
            alpha_gripper = np.zeros(5)
            d_gripper = np.zeros(5)



            # MODIFIED  
            joint_angles_degrees_gripper[1]=1.83759# for left down
            joint_angles_degrees_gripper[2]=2.8658# for left up
            joint_angles_degrees_gripper[3]=1.30405# for right down
            joint_angles_degrees_gripper[4]=-2.8658# for right up

            if name=="issac2sim":
                a_gripper[0]=-0.0 # pass the gripper a value to the uniform kinematic
            elif name=="push_box":
                a_gripper[0]=-0.0
            elif name=="novel_pose":
                a_gripper[0]=-0.025 # gripper close case
            elif name=="grasp":
                a_gripper[0]=-0.025 # gripper close case
            elif name=="grasp_object":
                a_gripper[0]=-0.025 # gripper close case
            else:
                a_gripper[0]=-0.025 # gripper close case
            
            a_gripper[1]=-0.03853/scale_factor[2]# gt from phyiscs parameter
            a_gripper[2]=0.041998/scale_factor[2]
            # we apologize for this due to the numerical error, 
            #the scale is really a huge gap from meters to mm so minor error may raise, we will fixe it in future
            a_gripper[3]=0.03853*0.6
            a_gripper[4]=0.041998/scale_factor[2]

            
            alpha_gripper[1]=np.pi/2
            alpha_gripper[2]=0
            alpha_gripper[3]=np.pi/2
            alpha_gripper[4]=0

            d_gripper[0]=0.11035/scale_factor[1] # gt from phyiscs parameter
            # d_gripper[0]=0
            d_gripper[1]=0
            d_gripper[2]=0
            d_gripper[3]=0
            d_gripper[4]=0
    #n
    a = torch.tensor([0.12, -0.05, -0.05, 0.125, -0.2125, -0.2, 0.23
                      , 0,
                      0,0,0,0,0
                      ])
    alpha = torch.tensor([0, -torch.pi/2, -torch.pi/2, torch.pi/2, -torch.pi/2, torch.pi/2, torch.pi/2,
                          0,
                          0,0,0,0,0
                          ])
    #n-1
    d = torch.tensor([0.69, -0.08, -0.66, 0.1, -0.744, 0.14, 0, 
                      0.0,
                      0.0,0.0,0.0,0.0,0
                    ])


    transformations = []
    # Store the cumulative transformations from base up to each joint
    final_transformations_list = []

    # First handle joints 0 through 7 (arm)
    final_transformation = torch.eye(4)
    


    # for the arm part, which is from 0 to 7, follow this way

    for i in range(9):
        T_temp = create_transformation_matrix_mdh(thetas[i], a[i], alpha[i], d[i])
        T_temp = reflect_axis(T_temp)
        transformations.append(T_temp)  # store the raw MDH for each joint
        # Multiply cumulatively
        final_transformation = torch.mm(final_transformation, T_temp)
        final_transformations_list.append(final_transformation.clone())

    # Save the transform at joint 7
    T7 = final_transformation.clone()

    # Now handle the finger joints 8 through 11


    # the logic of gripper

    # 9,10 are connect to 8,

    # 11,12 are connect to 8, 
    for i in range(9, 12):
        T_temp = create_transformation_matrix_mdh(thetas[i], a[i], alpha[i], d[i])
        T_temp = reflect_axis(T_temp)
        transformations.append(T_temp)  
        # Each finger frame i is computed from T7, 
        # not from the previous finger frame
        T_finger = torch.mm(T7, T_temp)
        final_transformations_list.append(T_finger.clone())

    return transformations, final_transformations_list

def inverse_affine_transformation_torch(transforms):
    """
    Compute the inverse of affine transformation matrices using PyTorch.

    Args:
        transforms (torch.Tensor): A tensor of shape (N, 4, 4) representing N affine transformation matrices.

    Returns:
        torch.Tensor: A tensor of shape (N, 4, 4) representing the inverse transformation matrices.
    """
    inv_transforms = []
    for transform in transforms:
        M = transform[:3, :3]  # Extract the rotation matrix (3x3)
        b = transform[:3, 3]   # Extract the translation vector (3,)

        M_inv = torch.inverse(M)  # Compute the inverse of the rotation matrix
        b_new = -M_inv @ b        # Compute the inverse translation

        # Construct the inverse affine transformation matrix
        inv_A = torch.zeros((4, 4), device=transform.device, dtype=transform.dtype)
        inv_A[:3, :3] = M_inv
        inv_A[:3, 3] = b_new
        inv_A[3, 3] = 1

        inv_transforms.append(inv_A)

    return torch.stack(inv_transforms)

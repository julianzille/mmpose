dataset_info = dict(
    dataset_name='acino3d',
    paper_info=dict(
        author='Zille, Julian',
        title='AcinoSet: A 3D Pose Estimation Dataset and Baseline Models for Cheetahs in the Wild',
        container='2021 IEEE International Conference on Robotics and Automation',
        year='2022',
        homepage='https://github.com/African-Robotics-Unit/AcinoSet'
    ),

    # [B, G, R]

    keypoint_info={
        0:
            dict(
            name='Root',
            id=0,
            color=[0,100,0],
            type='upper',
            swap=''
        ),
        1:
        dict(
            name='R_Eye',
            id=1,
            color=[203,192,255],
            type='upper',
            swap='L_Eye'),
        2:
        dict(name='Nose', id=2, color=[203,192,255], type='upper', swap=''),
        3:
        dict(name='Neck', id=3, color=[51, 153, 255], type='upper', swap=''),
        4:
        dict(
            name='Root of tail',
            id=4,
            color=[51, 153, 255],
            type='lower'),
        5:
        dict(
            name='L_Shoulder',
            id=5,
            color=[0, 205, 255],
            type='upper',
            swap='R_Shoulder'),
        6:
        dict(
            name='L_Elbow',
            id=6,
            color=[0, 205, 255],
            type='upper',
            swap='R_Elbow'),
            
        7:
        dict(
            name='R_Shoulder',
            id=7,
            color=[6, 156, 250],
            type='upper',
            swap='L_Shoulder'),
        8:
        dict(
            name='R_Elbow',
            id=8,
            color=[6, 156, 250],
            type='upper',
            swap='L_Elbow'),
       
        9:
        dict(
            name='L_Hip',
            id=9,
            color=[0, 205, 255],
            type='lower',
            swap='R_Hip'),
        10:
        dict(
            name='L_Knee',
            id=10,
            color=[0, 205, 255],
            type='lower',
            swap='R_Knee'),
    
        
        11:
        dict(
            name='R_Hip', id=11, color=[6, 156, 250], type='lower',
            swap='L_Hip'),
        12:
        dict(
            name='R_Knee',
            id=12,
            color=[6, 156, 250],
            type='lower',
            swap='L_Knee'),
    
        
        13:
        dict(
            name='tail_tip',
            id=13,
            color=[128,0,0],
            type='lower'),
        14:
        dict(
            name='tail_mid',
            id=14,
            color=[128,0,0],
            type='lower'
        ),
        15:
        dict(
            name='r_front_ankle',
            id=15,
            color=[6, 156, 250],
            type='upper',
            swap='l_front_ankle'
        ),
        16:
        dict(
            name='l_front_ankle',
            id=16,
            color=[0,205,255],
            type='upper',
            swap='r_front_ankle'
        ),
        17:
        dict(
            name='r_back_ankle',
            id=17,
            color=[6, 156, 250],
            type='lower',
            swap='l_back_ankle'
        ),
        18:
        dict(
            name='l_back_ankle',
            id=18,
            color=[0,205,255],
            type='lower',
            swap='r_back_ankle'
        ),
        19:
        dict(
            name='L_Eye', id=19, color=[203,192,255], type='upper', swap='R_Eye')
        
        
    },
    skeleton_info={
        0: dict(link=('L_Eye', 'R_Eye'), id=0, color=[71, 99, 255]),
        1: dict(link=('L_Eye', 'Nose'), id=1, color=[71, 99, 255]),
        2: dict(link=('R_Eye', 'Nose'), id=2, color=[71, 99, 255]),
        3: dict(link=('Nose', 'Neck'), id=3, color=[71, 99, 255]),
        4: dict(link=('Neck', 'Root'), id=4, color=[50, 205, 50]),
        5: dict(link=('Neck', 'L_Shoulder'), id=5, color=[0, 255, 255]),
        6: dict(link=('L_Shoulder', 'L_Elbow'), id=6, color=[0, 255, 255]),
        7: dict(link=('L_Elbow', 'l_front_ankle'), id=7, color=[0, 255, 255]),
        8: dict(link=('Neck', 'R_Shoulder'), id=8, color=[6, 156, 250]),
        9: dict(link=('R_Shoulder', 'R_Elbow'), id=9, color=[6, 156, 250]),
        10: dict(link=('R_Elbow', 'r_front_ankle'), id=10, color=[6, 156, 250]),
        11: dict(link=('Root of tail', 'L_Hip'), id=11, color=[0, 255, 255]),
        12: dict(link=('L_Hip', 'L_Knee'), id=12, color=[0, 255, 255]),
        13: dict(link=('L_Knee', 'l_back_ankle'), id=13, color=[0, 255, 255]),
        14: dict(link=('Root of tail', 'R_Hip'), id=14, color=[6, 156, 250]),
        15: dict(link=('R_Hip', 'R_Knee'), id=15, color=[6, 156, 250]),
        16: dict(link=('R_Knee', 'r_back_ankle'), id=16, color=[6, 156, 250]),
        17: dict(link=('Root','Root of tail'), id=17, color=[50, 205, 50]),
        18:dict(link=('tail_mid','tail_tip'),id=18,color=[225,65,105]),
        19:dict(link=('Root of tail','tail_mid'),id=19,color=[225,65,105])
    },
    joint_weights=[
        1., 1., 1., 1., 1., 1., 1., 1.2, 1.2, 1.2, 1.2, 1., 1., 1.2, 1.2, 1.2,
        1.2, 1.2, 1.2, 1.2
    ],
    sigmas=[
    .26, .25, .25, .35, .35, .79, .79, .72, .72, .62, .62, 1.07, 1.07,
    .87, .87, .89, .89, .35, .35, .79
    ], # https://github.com/cocodataset/cocoapi/issues/399,
    
    stats_info=dict(bbox_center=(1269.,  760.), bbox_scale=1.6)
)

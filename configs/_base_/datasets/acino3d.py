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
            dict(name='Nose', id=0, color=[203,192,255], type='upper', swap=''),
        1:
            dict(name='R_Eye',id=1,color=[203,192,255],type='upper',swap='L_Eye'),
        2:
            dict(name='L_Eye', id=2, color=[203,192,255], type='upper', swap='R_Eye'),
        3:
            dict(name='Neck', id=3, color=[51, 153, 255], type='upper', swap=''),

        4:
            dict(name='Spine',id=4,color=[0,100,0],type='upper',swap=''),
        5:
            dict(name='tail_root',id=5,color=[51, 153, 255]),
        6:
            dict(name='tail_mid',id=6,color=[128,0,0]),
        7:
            dict(name='tail_tip',id=7,color=[128,0,0]),

        8:
            dict(name='R_Shoulder',id=8,color=[6, 156, 250],type='upper',swap='L_Shoulder'),
        9:  
            dict(name='R_Front_Knee',id=9,color=[6, 156, 250],swap='L_Front_Knee'),
        10:
            dict(name='R_Front_Ankle',id=10,color=[6, 156, 250],type='lower',swap='L_Front_Ankle'),
        
        11:
            dict(name='L_Shoulder',id=11,color=[0, 205, 255],type='upper',swap='R_Shoulder'),  
        12:
            dict(name='L_Front_Knee',id=12,color=[0, 205, 255],swap='R_Front_Knee'),
        13:
            dict(name='L_Front_Ankle',id=13,color=[0,205,255],type='lower',swap='R_Front_Ankle'),
        
        14:
            dict(name='R_Hip', id=14, color=[6, 156, 250],swap='L_Hip'),
        15:
            dict(name='R_Back_Knee',id=15,color=[6, 156, 250],swap='L_Back_Knee'),
        16:
            dict(name='R_Back_Ankle',id=16,color=[6, 156, 250],type='lower',swap='L_Back_Ankle'),
        
        17:
            dict(name='L_Hip',id=17,color=[0, 205, 255],swap='R_Hip'),
        18: 
            dict(name='L_Back_Knee',id=18,color=[0, 205, 255],swap='R_Back_Knee'),
        19:
            dict(name='L_Back_Ankle',id=19,color=[0,205,255],type='lower',swap='R_Back_Ankle')    
    },
    skeleton_info={
        0: dict(link=('L_Eye', 'R_Eye'), id=0, color=[71, 99, 255]),
        1: dict(link=('L_Eye', 'Nose'), id=1, color=[71, 99, 255]),
        2: dict(link=('R_Eye', 'Nose'), id=2, color=[71, 99, 255]),
        3: dict(link=('Nose', 'Neck'), id=3, color=[71, 99, 255]),
        4: dict(link=('Neck', 'Spine'), id=4, color=[50, 205, 50]),
        5: dict(link=('Neck', 'L_Shoulder'), id=5, color=[0, 255, 255]),
        6: dict(link=('L_Shoulder', 'L_Front_Knee'), id=6, color=[0, 255, 255]),
        7: dict(link=('L_Front_Knee', 'L_Front_Ankle'), id=7, color=[0, 255, 255]),
        8: dict(link=('Neck', 'R_Shoulder'), id=8, color=[6, 156, 250]),
        9: dict(link=('R_Shoulder', 'R_Front_Knee'), id=9, color=[6, 156, 250]),
        10: dict(link=('R_Front_Knee', 'R_Front_Ankle'), id=10, color=[6, 156, 250]),
        11: dict(link=('tail_root', 'L_Hip'), id=11, color=[0, 255, 255]),
        12: dict(link=('L_Hip', 'L_Back_Knee'), id=12, color=[0, 255, 255]),
        13: dict(link=('L_Back_Knee', 'L_Back_Ankle'), id=13, color=[0, 255, 255]),
        14: dict(link=('tail_root', 'R_Hip'), id=14, color=[6, 156, 250]),
        15: dict(link=('R_Hip', 'R_Back_Knee'), id=15, color=[6, 156, 250]),
        16: dict(link=('R_Back_Knee', 'R_Back_Ankle'), id=16, color=[6, 156, 250]),
        17: dict(link=('Spine','tail_root'), id=17, color=[50, 205, 50]),
        18:dict(link=('tail_mid','tail_tip'),id=18,color=[225,65,105]),
        19:dict(link=('tail_root','tail_mid'),id=19,color=[225,65,105])
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

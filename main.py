import cv2
import numpy as np

from parameters import Parameters
from data_retrieval import Retrieve_data
from optimizer import Optimizer
from evaluator import Evaluate


class Hybrid_optimization:

    def __init__(self, data, params):
        self.data = data
        self.params = params
        self.kf_factor = 6
        self.use_kf_based = True
        self.use_initSol = False
        self.writeOutput2file = False
        self.prev_data = None
        self.kf_poses = {}
        self.poses = []
        self.positions = []


        # Optimization method
        # de: differential evolution
        # pso: particle swarm optimization
        # sade: self adaptive DE
        self.optMethod = 'de'

        # Objective function list
        self.optimizationFuncs = {'semantic_reproj_loss':False,
                                  'non_semantic_reproj_loss':True,
                                  'semSize_consistency_loss':False,
                                  'place_recognition_loss':False,
                                  'point_in_semantic_reproj_loss':False}

        self.performOptz = True if True in self.optimizationFuncs.values() else False

        # Generate random keyframes
        self.get_kfList()


    def run(self):
        self.org_path = []
        self.optz = Optimizer(self.optMethod, self.params, self.use_initSol,
                         self.optimizationFuncs, self.writeOutput2file)

        # start and end frame number: Required for evaluation as well
        self.sf = 100; self.ef = 105
        # Main Loop
        for i,dat in enumerate(self.data.gData[self.sf:self.ef]):
            print("\nFrame_ID->{}".format(dat['frame_ID']))
            pose_vec = self.get_poseVec(dat['pose'])
            place_pose = None

            # skip optimization step if first frame
            if i == 0:
                print('...Keyframe\nOriginal pose vector:', pose_vec)
                self.kf_poses[i] = {'sptam':dat['pose'], 'ho':pose_vec}
                self.org_path.append(dat['position'])
                #Store estimated pose obtained by S-PTAM
                self.poses.append(dat['pose'])
                self.positions.append(self.mat2position(self.poses[i]))
                continue

            # Check if place recognition is activated
            if self.performOptz:
                if dat['place_id'] is not None and dat['valid_place']:
                    print("Optimizing...")
                    place_pose = self.data.placePose_data[dat['place_id']]
                    self.optz.run(dat,self.kf_poses,place_pose)
                    print('Best solution:',np.round(self.optz.best_sol,decimals=5))
                    pose = self.vec2mat(np.round(self.optz.best_sol,decimals=5))
                    self.poses.append(pose)
                else:
                    #Store estimated pose obtained by S-PTAM
                    self.poses.append(dat['pose'])

                # Check if current frame is keyframe
                if self.is_keyframe(i):
                    #print('...Keyframe')
                    if self.optz.best_sol is not None:
                        self.kf_poses[i] = {'sptam':dat['spose'],'ho':self.optz.best_sol}
                    else:
                        self.kf_poses[i] = {'sptam':dat['pose'],'ho':pose_vec}
            # SPTAM
            else:
                #Store estimated pose obtained by S-PTAM
                self.poses.append(dat['pose'])

            # Store data
            self.org_path.append(dat['position'])
            pose_vec = self.get_poseVec(dat['pose'])
            self.positions.append(self.mat2position(self.poses[i]))
            print('Original pose vector:', pose_vec)
            print('--Positions:\nsptam',dat['position'])
            print('recomputed',self.positions[i])
        print('-------------------------------------------------------------------------\n')


    # Generate random keyframe indices
    def get_kfList(self):
        self.kf_list = []
        datSize = len(self.data.gData)
        generate_index = True
        j = 1

        if self.use_kf_based:
            for i in range(datSize):
                if i == 0:
                    self.kf_list.append(i)

                # get next keyframe index randomly
                if generate_index:
                    next_kf = np.random.randint(1,self.kf_factor)

                if j == next_kf:
                    j = 1
                    self.kf_list.append(i)
                    generate_index = True
                else:
                    generate_index = False
                    j += 1
        else:
            self.kf_list = [i for i in range(datSize)]


    def get_poseVec(self, pose):
        rmat = pose[:3,:3]
        tvec = -pose[:3,3:].reshape(3,)
        rvec = self.optz.get_rvec(rmat)
        return np.hstack((rvec, tvec))


    def vec2mat(self, vec):
        rvec = np.array(vec[:3])
        tvec = np.array(vec[3:])
        rmat, Jocobian = cv2.Rodrigues(rvec)
        pose = np.hstack((rmat, tvec.reshape(3,1)))
        return np.vstack((pose, [0., 0., 0., 1.]))


    def mat2position(self, pose):
        rmat = pose[:3, :3]
        tvec = pose[:3,3:].reshape(3,)
        return -rmat.T.dot(tvec)


    def is_keyframe(self,i):
        return True if i in self.kf_list else False



if __name__ == '__main__':
    # Observations file
    obsv_file = open("../loss_files/loss_1.txt","r")
    # Semantic database file
    dB_file = "../loss_files/place_semantic.json"
    # Pose file
    pose_file = open("../loss_files/Positions_v2.txt","r")
    # Raw SPTAM pose file
    sptam_file = open("../loss_files/raw_sptam_Positions.txt","r")
    # Place pose file
    placePose_file = open('../loss_files/place_pose.txt','r')
    # non-semantic point observation file
    ptObs_file = open("../loss_files/all_observations.txt","r")
    # Ground truth file
    gt_file = open("../../ground_truth/trajectory_22.txt")
    # map contour
    map_contour = cv2.imread("../../ground_truth/map_contour.png")

    params = Parameters(map_contour)

    data = Retrieve_data(params,obsv_file,dB_file,pose_file,
                            sptam_file,placePose_file,ptObs_file)
    data.store()

    hybOpt = Hybrid_optimization(data,params)
    hybOpt.run()

    # Perform evaluation
    eval = Evaluate(gt_file, hybOpt.org_path,hybOpt.positions,
                                        [hybOpt.sf, hybOpt.ef])
    eval.eval()

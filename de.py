#imports
from os import mkdir
import math
from statistics import median, stdev
from matplotlib import pyplot as plt
import time as t
from time import gmtime, strftime, time
from random import uniform, choice, randint
import uuid
from tqdm import tqdm

import cv2
import numpy as np

class DE:

    def __init__(self, params, opt_funcs, data, kf_poses, place_pose, write2file):
        self.pop = [] #population's positions
        self.m_nmdf = 0.00 #diversity variable
        self.diversity = []
        self.fbest_list = []

        self.params = params
        self.opt_funcs = opt_funcs
        self.data = data
        self.kf_poses = kf_poses
        self.place_pose = place_pose
        self.write2file = write2file


    def generateGraphs(self, fbest_list, diversity_list, max_iterations, uid, run):
        plt.plot(range(0, max_iterations), fbest_list, 'r--')
        if self.write2file:
            plt.savefig(str(uid) + '/graphs/run' + str(run) + '_' + 'convergence.png')
        plt.clf()
        plt.plot(range(0, max_iterations), diversity_list, 'b--')
        if self.write2file:
            plt.savefig(str(uid) + '/graphs/run' + str(run) + '_' + 'diversity.png')
        plt.clf()


    def updateDiversity(self):
        diversity = 0
        aux_1 = 0
        aux2 = 0
        a = 0
        b = 0
        d = 0


        for a in range(0, len(self.pop)):
            b = a+1
            for i in range(b, len(self.pop)):
                aux_1 = 0

                ind_a = self.pop[a]
                ind_b = self.pop[b]

                for d in range(0, len(self.pop[0])):
                    aux_1 = aux_1 + (pow(ind_a[d] - ind_b[d], 2).real)
                aux_1 = (math.sqrt(aux_1).real)
                aux_1 = (aux_1 / len(self.pop[0]))

                if b == i or aux_2 > aux_1:
                    aux_2 = aux_1
            diversity = (diversity) + (math.log((1.0) + aux_2).real)

        if self.m_nmdf < diversity:
            self.m_nmdf = diversity

        return (diversity/self.m_nmdf).real


    #fitness_function
    def fitness(self, individual):
        rotWeight = 1
        placeWeight = 10
        pixelWeight = 0.01
        lenWeight = 1
        maxPixelError = 1000

        rvec = np.array(individual[:3])
        tvec = np.array(individual[3:])
        rvec[0] = 0.0
        rvec[2] = 0.0
        tvec[1] = 0.0
        rmat, Jocobian = cv2.Rodrigues(rvec)
        soe = None

        # Get camera projection matrix
        ext_mat = np.hstack((rmat, tvec.reshape(3,1)))
        #proj_mat = self.params.K_mat.dot(ext_mat)
        ext_mat_s = np.vstack((ext_mat, [0., 0., 0., 1.]))
        ext_mat_s = np.linalg.inv(ext_mat_s)
        ext_mat_s = ext_mat_s[:3,]
        proj_mat = self.params.K_mat.dot(ext_mat_s)

        is_valid = True
        # First check if  individual is within valid workspace
        if self.params.use_workspace_validator:
            t = tvec.copy()
            h,w = self.params.map_contour.shape[0:2]
            position = -rmat.T.dot(t.reshape(3,))
            x = int(position[0]/self.params.scale) + int(h/2)
            z = int(position[2]/self.params.scale) + int(w/2)

            not_in_img = (x>=h or z>=w) or (x<0 or z<0 )

            if not_in_img or self.params.map_contour[x][z][0] == 0:
                soe = np.inf
                #soe = 99999999999.

        # LOSS-1: Semantic corner reprojection loss
        if self.opt_funcs['semantic_reproj_loss'] and self.data['point_data'] is not None and len(self.data['point_data']['points2d'])>0:
            # 2D and 3D points of semantic corners from observarions
            pt2d = self.data['point_data']['points2d'].copy()
            pt3d = self.data['point_data']['points3d'].copy()

            # If SPTAM's coord-system is to be used TODO: Verify
            pt3d = [pt3d[0], pt3d[1], pt3d[2], pt3d[3]] #RK changes

            e1 = 0. # reprojection error
            for p2,p3 in zip(pt2d,pt3d):
                P = p3.copy()
                P[1] = -P[1]
                P = np.append(P,1.)
                proj_pt = np.dot(proj_mat, P)
                if proj_pt[2] <= 0:
                    err = maxPixelError ## PARAM ALERT
                else:
                    proj_pt = proj_pt[:2]/proj_pt[2]
                    err = proj_pt - np.array(p2).reshape(2,)
                    err = abs(err[0]) + abs(err[1])
                e1 += err
            if soe is None:
                soe = pixelWeight*min(e1,maxPixelError)
            else:
                soe += pixelWeight*min(e1,maxPixelError)

        # LOSS-2: Non-semantic loss
        if self.opt_funcs['non_semantic_reproj_loss']:
            # reference keyframe
            ref_kf = max(list(self.kf_poses.keys()))

            if self.kf_poses[ref_kf]['ho'] is not None:
                # pose vector from hybrid-optimizer
                rvec_kf = np.array(self.kf_poses[ref_kf]['ho'][:3])
                tvec_kf = np.array(self.kf_poses[ref_kf]['ho'][3:])

                # Get pose of the referencing keyframe
                rmat_kf, _ = cv2.Rodrigues(rvec_kf)
                pose_kf = np.hstack((rmat_kf, tvec_kf.reshape(3,1)))
                pose_kf = np.vstack((pose_kf, [0.,0.,0.,1.]))

                # Translation vector between KF and current frame
                cur_pose = np.vstack((ext_mat,[0.,0.,0.,1.]))
                objtvPose1 = np.linalg.inv(cur_pose).dot(pose_kf)
                R1 = objtvPose1[:3,:3]# rotation matrix
                t1 = objtvPose1[:3,3:].reshape(3,)
                local_tvec1 = -R1.T.dot(t1) # TO VERIFY: verify if the vector suffices

                # SPTAM translation vector
                objtvPose2 = np.linalg.inv(self.data['spose']).dot(\
                                            self.kf_poses[ref_kf]['sptam'])
                R2 = objtvPose2[:3,:3]# rotation matrix
                t2 = objtvPose2[:3,3:].reshape(3,)
                local_tvec2 = -R2.T.dot(t2)

                # difference in translation error
                #e2 = np.absolute(local_tvec1 - local_tvec2)
                R1,_ = cv2.Rodrigues(R1)
                R2,_ = cv2.Rodrigues(R2)
                e2 = np.linalg.norm(t1 - t2) + rotWeight*np.linalg.norm(R1 - R2) #RK this and following
                #e2 = np.linalg.norm(t1 - t2) + rotWeight*np.linalg.norm(self.get_rvec(R1)-self.get_rvec(R2))

                if soe is not None:
                    soe += e2
                    #soe += sum(e2)
                else:
                    soe = e2
                    #soe = sum(e2)
            else:
                print('**')

        # LOSS-3: Semantic size consistency loss
        if self.opt_funcs['semSize_consistency_loss'] and self.data['point_data'] is not None and len(self.data['point_data']['points2d'])>0:
            # 2D and 3D points of semantic corners from database
            pt2DB = self.data['point_data']['db_points2d'].copy()
            pt2Obs = self.data['point_data']['points2d'].copy()
            pt3d = self.data['point_data']['points3d'].copy()

            try:
                pt2DB = pt2DB.reshape(int(len(pt2DB)/4),4,2)
                pt2Obs = pt2Obs.reshape(int(len(pt2Obs)/4),4,2)
                pt3d = pt3d.reshape(int(len(pt3d)/4),4,3)
                compute_loss3 = True
            except:
                print("Not enough corners for loss-3")
                compute_loss3 = False

            if compute_loss3:
                db_fx = self.params.db_intrinsics[0]
                obs_fx = self.params.intrinsics[0]

                e3 = []
                for p2DB, p2Obs, p3 in zip(pt2DB, pt2Obs, pt3d):
                    # top_left
                    db_tl = p2DB[0]
                    ptl = np.append(p3[0],1.)
                    ptl[1] = -ptl[1]
                    proj_tl = np.dot(proj_mat,ptl)
                    if proj_tl[2]<=0:
                        e3.append(1000) #RK Param Alert
                    scale_tl = proj_tl[2]
                    proj_tl = proj_tl[:2]/proj_tl[2]
                    obs_tl = p2Obs[0]

                    # top_right
                    db_tr = p2DB[1]
                    ptr = np.append(p3[1],1.)
                    ptr[1] = -ptr[1]
                    proj_tr = np.dot(proj_mat,ptr)
                    if proj_tr[2]<=0:
                        e3.append(1000) #RK Param Alert
                    scale_tr = proj_tr[2]
                    proj_tr = proj_tr[:2]/proj_tr[2]
                    obs_tr = p2Obs[1]


                    # bottom_right
                    db_br = p2DB[3]
                    pbr = np.append(p3[3],1.)
                    pbr[1] = -pbr[1]
                    proj_br = np.dot(proj_mat,pbr)
                    if proj_br[2]<=0:
                        e3.append(1000) #RK Param Alert
                    scale_br = proj_br[2]
                    proj_br = proj_br[:2]/proj_br[2]
                    obs_br = p2Obs[3]

                    if proj_tl[0]>proj_br[0] or proj_tl[1]>proj_br[1]:
                        e3.append(1000) #RK Param Alert

                    # Observation data
                    obs_len = ((scale_tl+scale_tr)/2.0) * np.linalg.norm((obs_tl,obs_tr))/obs_fx
                    obs_width = ((scale_tr+scale_br)/2.0) * np.linalg.norm((obs_tr,obs_br))/obs_fx
                    obs_diag = ((scale_tl+scale_br)/2.0) *np.linalg.norm((obs_tl,obs_br))/obs_fx

                    # dB data
                    db_len = np.linalg.norm(p3[0]-p3[1])
                    db_width = np.linalg.norm(p3[1]-p3[3])
                    db_diag = np.linalg.norm(p3[0]-p3[3])

                    # Differences
                    diff_len = np.absolute(obs_len - db_len)
                    diff_width = np.absolute(obs_width - db_width)
                    diff_diag = np.absolute(obs_diag - db_diag)

                    # Error
                    e3.append(diff_len + diff_width + diff_diag)

                if soe is not None:
                    soe += lenWeight*np.sum(e3)
                else:
                    soe = lenWeight*np.sum(e3)
                # or use soe += sum(e3)
            else:
                print("proceeding without computing loss-3")

            # LOSS-4: place recognition loss

        # LOSS-4: place recognition loss
        if self.opt_funcs['place_recognition_loss']:
            if self.place_pose is None:
                pass #print("No place pose data")
            else:
                cur_pose = np.hstack((rvec,tvec))
                position = tvec
                orientation = rvec
                place_position = self.place_pose[:3] #RK Change
                place_orientation = self.place_pose[3:] #RK Change

                # SE2
                if self.params.use_SE2:
                    # X and Z values for SE2, indices at 0 and 2
                    position = position[[0,2]]
                    orientation = orientation[1] # rotation along y-axis
                    place_position = place_position[[0,2]]
                    place_orientation = place_orientation[1]
                    ang_threshold = self.params.place_angularThreshold[1]

                    position_diff = np.linalg.norm(position - place_position)\
                                                    - self.params.place_radius
                    orientation_diff = abs(place_orientation - orientation)

                    large_orientationError = True if orientation_diff > ang_threshold\
                                             else False

                    # Combine comaprison
                    # TO CHECK: Verify if the error should be zero if pose is within
                    # ...the place radius
                    e4 = abs(position_diff) + orientation_diff\
                         if position_diff > 0 or large_orientationError else 0. # TODO: multiply weight to orientation

                    '''
                    # Seperate comparison
                    e4 = position_diff if position_dif > 0. else 0.
                    e4 = e4 + (sum(orientation_diff) if large_orientationError else 0.)'''

                # SE3
                else: # RK CHANGED BELOW LINES
                    position_diff = np.linalg.norm(position - place_position)\
                                                    - self.params.place_radius
                    orientation_diff = [math.atan2(math.sin(place_orientation[i] - orientation[i]),math.cos(place_orientation[i] - orientation[i])) for i in range(0,3)]
                    orientation_diff = np.abs(orientation_diff)

                    large_orientationError = True in [True for a,b in \
                            zip(orientation_diff, self.params.place_angularThreshold)\
                            if a > b]

                    e4 = position_diff if position_diff > 0 else 0.
                    e4 = e4 + rotWeight*(sum(orientation_diff) if large_orientationError else 0.)

                if soe is None:
                    soe = placeWeight*e4
                else:
                    soe += (placeWeight*e4)

        # LOSS-5: Points in semantic re-projection loss
        # TO CHECK: What if the error in this loss-function is zero?
        if self.opt_funcs['point_in_semantic_reproj_loss']:
            e5 = 0.
            for sem in self.data['point_data']['semantic_obsvervation']:
                box_corners = list(sem.values())[0]

                # NOTE: point storage format: <width,height>
                tl = box_corners[0] # top left
                br = box_corners[3] # bottom right

                observations = self.data['point_inBox']['point_data']
                side_list = observations[0]
                pt2d = observations[1]; pt3d = observations[2]

                for side, p2, p3 in zip(side_list,pt2d,pt3d):
                    # If point is inside the box
                    if (p2[0] > tl[0] and p2[0] < br[0]) and\
                    (p2[1] > tl[1] and p2[1] < br[1]):
                        # check if re-projected point is in the box
                        P = p3.copy()
                        P = np.append(P,1.)
                        proj_pt = np.dot(proj_mat, P)
                        if proj_pt[2] == 0:
                            e5 += 1000 ## PARAM ALERT
                            continue

                        proj_pt = proj_pt[:2]/proj_pt[2]
                        # Along width
                        if proj_pt[0] < tl[0]:
                            e5 += tl[0] - proj_pt[0]
                        elif proj_pt[0] > br[0]:
                            e5 +=  proj_pt[0] - br[0]

                        # Along height
                        if proj_pt[1] < tl[1]:
                            e5 += tl[1] - proj_pt[1]
                        elif proj_pt[1] > br[1]:
                            e5 +=  proj_pt[1] - br[1]

            if soe == 0.:
                pass
            elif soe is None:
                soe = e5
            else:
                soe += e5


        if soe is None:
            print("No data to compute objective function")
            sys.exit()

        return soe


    # Function to convert rotation matrix into rotation vector
    @staticmethod
    def get_rvec(rmat):
        # TODO: Use numpy instead of math
        theta = math.acos(((rmat[0,0]+rmat[1,1]+rmat[2,2]) - 1) / 2)
        sin_theta = math.sin(theta)
        if sin_theta == 0:
            rx, ry, rz = 0.0, 0.0, 0.0
        else:
            multi = 1 / (2 * math.sin(theta))
            rx = multi * (rmat[2, 1] - rmat[1, 2]) * theta
            ry = multi * (rmat[0, 2] - rmat[2, 0]) * theta
            rz = multi * (rmat[1, 0] - rmat[0, 1]) * theta
        return np.array([rx, ry, rz])


    def generatePopulation(self, pop_size, dim, bounds):
        ref_kf = max(list(self.kf_poses.keys())) #RK from here
        vo = np.linalg.inv(self.kf_poses[ref_kf]['sptam']).dot(self.data['spose'])

        rvec_kf = np.array(self.kf_poses[ref_kf]['ho'][:3])
        tvec_kf = np.array(self.kf_poses[ref_kf]['ho'][3:])
        rmat_kf, _ = cv2.Rodrigues(rvec_kf)
        pose_kf = np.hstack((rmat_kf, tvec_kf.reshape(3,1)))
        pose_kf = np.vstack((pose_kf, [0.,0.,0.,1.]))
        cur = pose_kf.dot(vo)
        rmat = cur[:3,:3]
        rvec,_ = cv2.Rodrigues(rmat)
        lp = []
        lp.append(rvec[0][0])
        lp.append(rvec[1][0])
        lp.append(rvec[2][0])
        lp.append(cur[0,3])
        lp.append(cur[1,3])
        lp.append(cur[2,3])
        self.pop.append(lp)

        for ind in range(pop_size):
            lp = []
            for d in range(dim):
                lp.append(uniform(bounds[d][0],bounds[d][1]))
            self.pop.append(lp)


    def evaluatePopulation(self):
        fpop = []
        for ind in self.pop:
            fpop.append(self.fitness(ind))
        return fpop


    def getBestSolution(self, maximize, fpop):
        fbest = fpop[0]
        best = [values for values in self.pop[0]]
        for ind in range(1,len(self.pop)):
            if maximize == True:
                if fpop[ind] >= fbest:
                    fbest = float(fpop[ind])
                    best = [values for values in self.pop[ind]]
            else:
                if fpop[ind] <= fbest:
                    fbest = float(fpop[ind])
                    best = [values for values in self.pop[ind]]

        return fbest,best


    def rand_1_bin(self, ind, dim, wf, cr):
        p1 = ind
        while(p1 == ind):
            p1 = choice(self.pop)
        p2 = ind
        while(p2 == ind or p2 == p1):
            p2 = choice(self.pop)
        p3 = ind
        while(p3 == ind or p3 == p1 or p3 == p2):
            p3 = choice(self.pop)

        # print('current: %s\n' % str(ind))
        # print('p1: %s\n' % str(p1))
        # print('p2: %s\n' % str(p2))
        # print('p3: %s\n' % str(p3))
        # input('...')

        cutpoint = randint(0, dim-1)
        candidateSol = []

        # print('cutpoint: %i' % (cutpoint))
        # input('...')

        for i in range(dim):
            if(i == cutpoint or uniform(0,1) < cr):
                candidateSol.append(p3[i]+wf*(p1[i]-p2[i])) # -> rand(p3) , vetor diferença (wf*(p1[i]-p2[i]))
            else:
                candidateSol.append(ind[i])

        # print('candidateSol: %s' % str(candidateSol))
        # input('...')
        # print('\n\n')
        return candidateSol


    def currentToBest_2_bin(self, ind, best, dim, wf, cr):
        p1 = ind
        while(p1 == ind):
            p1 = choice(self.pop)
        p2 = ind
        while(p2 == ind or p2 == p1):
            p2 = choice(self.pop)

        # print('current: %s\n' % str(ind))
        # print('p1: %s\n' % str(p1))
        # print('p2: %s\n' % str(p2))
        # input('...')

        cutpoint = randint(0, dim-1)
        candidateSol = []

        # print('cutpoint: %i' % (cutpoint))
        # input('...')

        for i in range(dim):
            if(i == cutpoint or uniform(0,1) < cr):
                candidateSol.append(ind[i]+wf*(best[i]-ind[i])+wf*(p1[i]-p2[i])) # -> rand(p3) , vetor diferença (wf*(p1[i]-p2[i]))
            else:
                candidateSol.append(ind[i])

        # print('candidateSol: %s' % str(candidateSol))
        # input('...')
        # print('\n\n')
        return candidateSol


    def boundsRes(self, ind, bounds):
        for d in range(len(ind)):
            if ind[d] < bounds[d][0]:
                ind[d] = bounds[d][0]
            if ind[d] > bounds[d][1]:
                ind[d] = bounds[d][1]


    def diferentialEvolution(self, pop_size, dim, bounds, max_iterations, runs, weight_factor=0.8, crossover_rate=0.9, maximize=True, operator=0):
        #generete execution identifier
        uid = uuid.uuid4()

        if self.write2file:
            mkdir(str(uid))
            mkdir(str(uid) + '/graphs')
            #to record the results
            results = open(str(uid) + '/results.txt', 'a')
            records = open(str(uid) + '/records.txt', 'a')

        if operator == 0:
            operatorStr = 'rand/1/bin'
        elif operator == 1:
            operatorStr = 'current to best/2/bin'

        if self.write2file:
            results.write('ID: %s\tDate: %s\tRuns: %s\tOperator: %s\n' % (str(uid ), strftime("%Y-%m-%d %H:%M:%S", gmtime()), str(runs), operatorStr))
            results.write('=================================================================================================================\n')
            records.write('ID: %s\tDate: %s\tRuns: %s\tOperator: %s\n' % (str(uid ), strftime("%Y-%m-%d %H:%M:%S", gmtime()), str(runs), operatorStr))
            records.write('=================================================================================================================\n')
        avr_fbest_r = []
        avr_diversity_r = []
        fbest_r = []
        best_r = []
        elapTime_r = []
        #runs
        for r in range(runs):
            elapTime = []
            start = time()
            if self.write2file:
                records.write('Run: %i\n' % r)
                records.write('Iter\tGbest\tAvrFit\tDiver\tETime\t\n')

            #start the algorithm
            best = [] #global best positions
            fbest = 0.00

            #global best fitness
            if maximize == True:
                fbest = 0.00
            else:
                fbest = math.inf

            #initial_generations
            self.generatePopulation(pop_size, dim, bounds)
            fpop = self.evaluatePopulation()

            # print('pop: %s\n' % str(self.pop))
            # print('fpop: %s\n' % str(fpop))

            fbest,best = self.getBestSolution(maximize, fpop)

            # print('fbest: %f\n' % (fbest))
            # print('best: %s\n' % str(best))
            # input('...')

            #evolution_step
            #for iteration in tqdm(range(max_iterations)):
            for iteration in range(max_iterations):
                avrFit = 0.00
                # #update_solutions
                for ind in range(0,len(self.pop)):
                    if operator == 0:
                        candSol = self.rand_1_bin(self.pop[ind], dim, weight_factor, crossover_rate)
                    elif operator == 1:
                        candSol = self.currentToBest_2_bin(self.pop[ind], best, dim, weight_factor, crossover_rate)

                    # print('candSol: %s' % str(candSol))

                    self.boundsRes(candSol, bounds)
                    fcandSol = self.fitness(candSol)

                    # print('candSolB: %s' % str(candSol))
                    # print('fcandSol: %f\n' % (fcandSol))

                    if maximize == False:
                        if fcandSol <= fpop[ind]:
                            self.pop[ind] = candSol
                            fpop[ind] = fcandSol
                    else:
                        if fcandSol >= fpop[ind]:
                            self.pop[ind] = candSol
                            fpop[ind] = fcandSol
                    avrFit += fpop[ind]
                avrFit = avrFit/pop_size
                self.diversity.append(self.updateDiversity())

                fbest,best = self.getBestSolution(maximize, fpop)

                self.fbest_list.append(fbest)
                elapTime.append((time() - start)*1000.0)
                if self.write2file:
                    records.write('%i\t%.4f\t%.4f\t%.4f\t%.4f\n' % (iteration, round(fbest,4), round(avrFit,4), round(self.diversity[iteration],4), elapTime[iteration]))
            if self.write2file:
                records.write('Pos: %s\n\n' % str(best))
            fbest_r.append(fbest)
            best_r.append(best)
            elapTime_r.append(elapTime[max_iterations-1])
            self.generateGraphs(self.fbest_list, self.diversity, max_iterations, uid, r)
            avr_fbest_r.append(self.fbest_list)
            avr_diversity_r.append(self.diversity)

            self.pop = []
            self.m_nmdf = 0.00
            self.diversity = []
            self.fbest_list = []

        #print("Best solution:", best)
        self.best_sol = best # TODO: Verify if this is the best solution

        fbestAux = [sum(x)/len(x) for x in zip(*avr_fbest_r)]
        diversityAux = [sum(x)/len(x) for x in zip(*avr_diversity_r)]
        self.generateGraphs(fbestAux, diversityAux, max_iterations, uid, 'Overall')
        if self.write2file:
            records.write('=================================================================================================================')
            if maximize==False:
                results.write('Gbest Overall: %.4f\n' % (min(fbest_r)))
                results.write('Positions: %s\n\n' % str(best_r[fbest_r.index(min(fbest_r))]))
            else:
                results.write('Gbest Overall: %.4f\n' % (max(fbest_r)))
                results.write('Positions: %s\n\n' % str(best_r[fbest_r.index(max(fbest_r))]))

            results.write('Gbest Average: %.4f\n' % (sum(fbest_r)/len(fbest_r)))
            results.write('Gbest Median: %.4f #probably should use median to represent due probably non-normal distribution (see Shapiro-Wilk normality test)\n' % (median(fbest_r)))
            if runs > 1:
                results.write('Gbest Standard Deviation: %.4f\n\n' % (stdev(fbest_r)))
            results.write('Elappsed Time Average: %.4f\n' % (sum(elapTime_r)/len(elapTime_r)))
            if runs > 1:
                results.write('Elappsed Time Standard Deviation: %.4f\n' % (stdev(elapTime_r)))
            results.write('=================================================================================================================\n')



'''
if __name__ == '__main__':
    from de import DE

    max_iterations = 100
    pop_size = 20
    dim = 2
    runs = 10
    bounds = ((-5.12,5.12), (-5.12,5.12))
    p = DE()
    p.diferentialEvolution(pop_size, dim, bounds, max_iterations, runs, maximize=False, operator=0)'''

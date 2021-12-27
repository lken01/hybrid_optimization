import ast
import json
import numpy as np


class Retrieve_data:

    def __init__(self,params,obsv_file,dB_file,pose_file,sptam_file,placePose_file,ptObs_file):
        self.obsv_file = obsv_file
        self.dB_file = dB_file
        self.pose_file = pose_file
        self.sptam_file = sptam_file
        self.placePose_file = placePose_file
        self.ptObs_file = ptObs_file
        self.scale = params.scale
        self.gData = None


    def store(self):
        # Read file contents
        obsv_lines = self.obsv_file.readlines()
        pose_lines = self.pose_file.readlines()
        sptam_lines = self.sptam_file.readlines()
        placePose_lines = self.placePose_file.readlines()
        pobs_lines = self.ptObs_file.readlines()
        with open(self.dB_file, "r") as f:
            dB_data = json.loads(f.read())
            f.close()

        # Store place pose data and charecteristic (place_ID & image_ID) data
        self.placePose_data = {}; characteristics = {}
        self.place_positions = []; self.place_ids = []
        for i,line in enumerate(placePose_lines):
            if i > 2:
                dat = line.strip('\n').split("  ")
                place_id = dat[0]
                pose = np.array(ast.literal_eval(dat[2]))
                angle_vec = np.array([0., 0., pose[3]]) # roll, pitch, yaw
                self.placePose_data[place_id] = np.hstack((pose[:-1]*self.scale,\
                                                                        angle_vec))
                characteristics[place_id] = dat[1]+".png"
                self.place_positions.append(pose[:-1]*self.scale)
                self.place_ids.append(place_id)

        # Store observarions, semantic SLAM pose and sptam pose data
        if len(obsv_lines) > len(pose_lines):
            pose_lines.insert(0,'"ID":0, None')
            sptam_lines.insert(0,'"ID":0, None')
            pobs_lines.insert(0,'None\n')

        assert len(obsv_lines)==len(pose_lines) and len(pose_lines)==len(sptam_lines),\
                                                                "Unequal data recorded"
        self.gData = [] # global data
        for i in range(len(obsv_lines)):
            if i == 0:
                # semantic SLAM position and pose
                position = np.zeros(3); pose = np.eye(4)
                # SPTAM position and pose
                sposition = position; spose = pose
                pobsDat = None
            else:
                # **Note that SPTAM pose requires inverse
                # semantic SLAM position and pose
                poseDat = json.loads(pose_lines[i])
                position = np.array(poseDat['Position'])
                pose = np.linalg.inv(np.array(poseDat['Matrix']))
                #pose = np.array(poseDat['Matrix'])

                # SPTAM position and pose
                sposeDat = json.loads(sptam_lines[i])
                sposition = np.array(sposeDat['Position'])
                spose = np.linalg.inv(np.array(sposeDat['Matrix']))
                #spose = np.array(sposeDat['Matrix'])

                #store non-semantic points
                pobsDat = json.loads(pobs_lines[i])

            obsvDat = json.loads(obsv_lines[i])
            place_id = obsvDat['place']
            # TODO: other heuristics can be added to check if place is valid
            valid_place = True if obsvDat["observations"] is not None and\
                                len(obsvDat["observations"]) > 0 else False

            frameDat = {'frame_ID': obsvDat['frame_ID'],
                        'place_id': place_id,
                        'pose': pose, 'position':position,
                        'spose': spose, 'sposition':sposition,
                        'valid_place': valid_place,
                        'point_data': None, 'point_inBox': pobsDat,}

            if obsvDat['place'] is not None:
                semDB_list = dB_data[characteristics[frameDat['place_id']]]

                # List of semantic names/labels in database corresponding to
                # ...the recognized place
                sdb_list = np.array([list(sdb.keys()) for sdb in semDB_list])
                sdb_list = list(sdb_list.flatten())

                ptDat = {}
                for obs, db in zip(obsvDat['observations'],obsvDat['db_points']):
                    pts2d = []; db_pts2d = [];pts3d = None
                    sem_name = db[0]

                    if sem_name in sdb_list:
                        # 2D points
                        for pt1, pt2 in zip(obs[1:],db[1:]):
                            # observation points
                            pts2d.append(pt1)
                            # dB points
                            db_pts2d.append(pt2)

                        # 3D points
                        for semDat in semDB_list:
                            if list(semDat.keys())[0] == sem_name:
                                pts3d = self.get_3Dpt(semDat, sem_name,
                                                        self.scale)

                        ptDat[sem_name] = {'points2d':pts2d,
                                           'db_points2d':db_pts2d,
                                           'points3d':pts3d}

                frameDat['point_data'] = self.reformat_data(ptDat)
            self.gData.append(frameDat)


    @staticmethod
    def get_3Dpt(data, sem_name, scale):
        points = []
        for pt in data[sem_name]:
            x, y, z = [float(p) for p in pt[1:]]
            points.append(np.array([x,y,z])*scale)
        return points


    @staticmethod
    def reformat_data(ptDat):
        pt2d = []; db2d = []; pt3d = []; semantic_data = []
        temp = {} #{'points2d':None,'db_points2d':None,'points3d':None}
        for semantic_key in ptDat:
            for pt2 in ptDat[semantic_key]['points2d']:
                pt2d.append(pt2)
            for dbPt in ptDat[semantic_key]['db_points2d']:
                db2d.append(dbPt)
            for pt3 in ptDat[semantic_key]['points3d']:
                pt3d.append(pt3)

            sem = {semantic_key: ptDat[semantic_key]['points2d']}
            semantic_data.append(sem)

        temp['points2d'] = np.array(pt2d)
        temp['db_points2d'] = np.array(db2d)
        temp['points3d'] = np.array(pt3d)
        temp['semantic_obsvervation'] = semantic_data

        return temp

import numpy as np
from matplotlib import pyplot as plt


class Evaluate:

    def __init__(self, gt_file, sptamPos, newPos, frame_range):
        self.gt_file = gt_file
        self.sptamPos = sptamPos
        self.newPos = newPos
        self.frame_range = frame_range
        self.source_inMap = [4000, 4000]
        self.scale = 0.01 # cm to m


    def eval(self):
        print('Evaluating...')
        self.pre_processGT()
        self.get_error()
        self.plot()


    def pre_processGT(self):
        self.gt_dat = []
        lines = self.gt_file.readlines()
        for line in lines:
            dat = (line.strip('\n')).split(', ')
            dat[1] = int(dat[1]) - 1
            dat[2] = (float(dat[2])-self.source_inMap[0]) * self.scale
            dat[3] = (float(dat[3])-self.source_inMap[1]) * self.scale
            self.gt_dat.append(dat)


    def get_error(self):
        self.error = []
        sse1 = 0; sse2 = 0 # sum of squared error
        for i,j in enumerate(range(self.frame_range[0],self.frame_range[1])):
            gt = np.array([self.gt_dat[j][3],self.gt_dat[j][2]])

            est = np.array([self.newPos[i][0],self.newPos[i][2]])
            err1 = np.linalg.norm((gt-est))
            sse1 += err1**2

            sptam = np.array([self.sptamPos[i][0],self.sptamPos[i][2]])
            err2 = np.linalg.norm((gt-sptam))
            sse2 += err2**2

        # no. of frames
        nof = self.frame_range[1] - self.frame_range[0]
        # root mean squared error
        rmse1 = np.sqrt(sse1/nof)
        rmse2 = np.sqrt(sse2/nof)

        print("Unit: Meters")
        print("RMSE of hybrid optimizer:", rmse1)
        print("RMSE of SPTAM+semantic SLAM:", rmse2)
        print("No. of frames compared", nof)
        print()

    def plot(self):
        plotAll_gt = True
        gtx = [p[3] for p in self.gt_dat]
        gtz = [p[2] for p in self.gt_dat]

        if not plotAll_gt:
            gtx = gtx[self.frame_range[0]:self.frame_range[1]]
            gtz = gtz[self.frame_range[0]:self.frame_range[1]]

        estx = [e[0] for e in self.newPos]
        estz = [e[2] for e in self.newPos]

        sptamx = [s[0] for s in self.sptamPos]
        sptamz = [s[2] for s in self.sptamPos]

        plt.style.use('fivethirtyeight')
        plt.tight_layout()
        plt.xlabel('x - axis')
        plt.ylabel('z - axis')
        plt.title('Evaluation')
        plt.plot(gtx,gtz,c='cornflowerblue',alpha=0.8,label='ground truth',linewidth=1)
        plt.plot(sptamx,sptamz,c='aquamarine',alpha=0.8,label='sem+sptam',linewidth=1)
        plt.plot(estx,estz,c='tomato', alpha=0.8,label='Hybrid optimization',linewidth=1)
        plt.legend()
        plt.show()

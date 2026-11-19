from os import path
import acoular as ac
import numpy as np

SrpGridX = 41
SrpGridY = 41

CameraWidth = 120.0
CameraHeight = 90.0
CameraDistance = 90.0

SpeedOfSound = 343.0  # Speed of sound in m/s

# Load UMA-16 microphone geometry
micgeofile = path.join(path.split(ac.__file__)[0], 'xml', 'minidsp_uma-16_mirrored.xml')
mg = ac.MicGeom(file=micgeofile)
CoordinatesOfMicrophones = []
for m in range(mg.num_mics):
    CoordinatesOfMicrophones.append(mg.pos[:, m])

class CGeometry:

    def __init__(self):
        self.__Delays = None

    def GetDelay(self, MicrofoneIndex1, MicrofoneIndex2, GridPositionX, GridPositionY):
        if self.__Delays is None:
            self.__Delays = np.zeros((len(CoordinatesOfMicrophones), len(CoordinatesOfMicrophones), SrpGridX, SrpGridY))

            for row in range(SrpGridY):
                y = row/SrpGridY * CameraHeight - CameraHeight/2
                for col in range(SrpGridX):
                    x = col/SrpGridX * CameraWidth - CameraWidth/2                            
                    Point = np.array([x, y, CameraDistance])
                    for m1 in range(len(CoordinatesOfMicrophones)):
                        d1 = np.linalg.norm(Point - CoordinatesOfMicrophones[m1])
                        for m2 in range(m1+1, len(CoordinatesOfMicrophones)):
                            d2 = np.linalg.norm(Point - CoordinatesOfMicrophones[m2])
                            DistanceDifference = d2 - d1
                            self.__Delays[m1, m2, row, col] = DistanceDifference / SpeedOfSound
                            self.__Delays[m2, m1, row, col] = -self.__Delays[m1, m2, row, col]
        return self.__Delays[MicrofoneIndex1, MicrofoneIndex2, GridPositionY, GridPositionX]

AGeometry = CGeometry()
import unittest
from pathlib import Path
import numpy as np
from python_scripts.simulation_metrics import get_destination_metrics

class Dimensions(unittest.TestCase):
    '''
    get_destination_metrics tests
    '''
    def test_dimensions(self):
        '''
        Testing the dimensions of the inputs: 
        making sure xpos and ypos are aligned and targetsx and targetsy are aligned
        '''
        xPos = np.array([50,50,50])
        yPos = np.array([50,60])
        targetsx = [40,50,60]
        targetsy = [70,70,70]
        with self.assertRaises(ValueError):
            reached, tsteps = get_destination_metrics(xPos, yPos, targetsx, targetsy)
        targetsx2 = [40,50]
        yPos_correct = np.array([50,60,70])
        with self.assertRaises(ValueError):
            reached, tsteps = get_destination_metrics(xPos, yPos_correct, targetsx2, targetsy)
        reached_cor, tsteps_cor = get_destination_metrics(xPos, yPos_correct, targetsx, targetsy)
        self.assertEqual(reached_cor, 1)
        self.assertEqual(tsteps_cor, 3)
        
class Reaching(unittest.TestCase):
    def test_reaching_5000(self):
        '''
        Testing the ability to detect when the agent doesn't reach a target 
        '''
        targetsx = [30,70]
        targetsy = [70,70]
        xPos = np.zeros(5000)
        yPos = np.zeros(5000)
        xPos[-1] = 30.1
        yPos[-1] = 69.9
        xPos_long = np.zeros(5001)
        yPos_long = np.zeros(5001)
        xPos_long[-1] = 30.1
        yPos_long[-1] = 69.9

        tar_5000, time_5000 = get_destination_metrics(xPos,yPos,targetsx,targetsy)
        tar_fail, time_fail = get_destination_metrics(xPos_long,yPos_long,targetsx,targetsy)
        tar_3000fail, time_3000 = get_destination_metrics(xPos,yPos,targetsx,targetsy,3000)
        tar_6000, time_6000 = get_destination_metrics(xPos_long,yPos_long,targetsx,targetsy,6000)

        self.assertEqual(0,tar_5000)
        self.assertEqual(5000, time_5000)
        self.assertEqual(-1,tar_fail)
        self.assertEqual(5001, time_fail)
        self.assertEqual(-1,tar_3000fail)
        self.assertEqual(3001, time_3000)
        self.assertEqual(0,tar_6000)
        self.assertEqual(5001, time_6000)


class SameDistance(unittest.TestCase):
    def test_samedistance(self):
        '''
        Testing when a target has been reached but the distance between two or more targets is the same
        '''
        targetsx = [30,50,70]
        targetsy = [70,70,70]
        xPos = np.array([50,45,42,40])
        yPos = np.array([50,57,64,70])
        targetsx2 = [50,50,50]
        targetsy2 = [70,70,70]
        xPos3 = np.array([50,50,50,45,42,40])
        yPos3 = np.array([50,60,65,67,69,70])
        targetsx3 = [0,40,40]
        targetsy3 = [0,70,70]
        tar_1, time_1 = get_destination_metrics(xPos,yPos,targetsx,targetsy)
        tar_2, time_2 = get_destination_metrics(xPos,yPos,targetsx2,targetsy2)
        tar_3, time_3 = get_destination_metrics(xPos3,yPos3,targetsx3,targetsy3)
        self.assertEqual(0,tar_1)
        self.assertEqual(4,time_1)
        self.assertEqual(0, tar_2)
        self.assertEqual(4, time_2)
        self.assertEqual(1, tar_3)
        self.assertEqual(6, time_3)

class Expected(unittest.TestCase):
    def test_expectectedcase(self):
        '''
        Testing expected 'normal' cases
        '''
        xy_reaches_3targpath = Path('tests') / 'test_inputs' / 'destination_metrics' / 'xypositions_3targ_reaches.csv'
        xy_fails_3targpath = Path('tests') / 'test_inputs' / 'destination_metrics' / 'xypositions_3targ_fails.csv'
        targ_3targpath = Path('tests') / 'test_inputs' / 'destination_metrics' / 'targetpositions_3targ.csv'
        xy_reaches_2targpath = Path('tests') / 'test_inputs' / 'destination_metrics' / 'xypositions_2targ_reaches.csv'
        xy_fails_2targpath = Path('tests') / 'test_inputs' / 'destination_metrics' / 'xypositions_2targ_fails.csv'
        targ_2targpath = Path('tests') / 'test_inputs' / 'destination_metrics' / 'targetpositions_2targ.csv'

        xy_reaches_3targ = np.loadtxt(xy_reaches_3targpath, delimiter=',')
        xy_fails_3targ = np.loadtxt(xy_fails_3targpath, delimiter=',')
        targ_3targ = np.loadtxt(targ_3targpath, delimiter=',')
        xy_reaches_2targ = np.loadtxt(xy_reaches_2targpath, delimiter=',')
        xy_fails_2targ = np.loadtxt(xy_fails_2targpath, delimiter=',')
        targ_2targ = np.loadtxt(targ_2targpath, delimiter=',')

        x3r = xy_reaches_3targ[0,:]
        y3r = xy_reaches_3targ[1,:]
        targx3 = targ_3targ[0,:]
        targy3 = targ_3targ[1,:]
        x3f = xy_fails_3targ[0,:]
        y3f = xy_fails_3targ[1,:]
        x2r = xy_reaches_2targ[0,:]
        y2r = xy_reaches_2targ[1,:]
        x2f = xy_fails_2targ[0,:]
        y2f = xy_fails_2targ[1,:]
        targx2 = targ_2targ[0,:]
        targy2 = targ_2targ[1,:]

        tar3r, time3r = get_destination_metrics(x3r,y3r,targx3,targy3)
        tar3f, time3f = get_destination_metrics(x3f,y3f,targx3,targy3)
        tar2r, time2r = get_destination_metrics(x2r,y2r,targx2,targy2)
        tar2f, time2f = get_destination_metrics(x2f,y2f,targx2,targy2,3000)

        self.assertEqual(1, tar3r)
        self.assertEqual(379, time3r)
        self.assertEqual(-1, tar3f)
        self.assertEqual(5001, time3f)
        self.assertEqual(1, tar2r)
        self.assertEqual(831, time2r)
        self.assertEqual(-1, tar2f)
        self.assertEqual(3001, time2f)

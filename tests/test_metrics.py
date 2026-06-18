import unittest
from pathlib import Path
import numpy as np
from python_scripts.simulation_metrics import get_destination_metrics
from python_scripts.simulation_metrics import get_success_rate

class DestinationMetricTests(unittest.TestCase):
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
        
    
    def test_reaching(self):
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


class SuccessRateTests(unittest.TestCase):
    '''
    get_succes_rate tests
    '''
    def test_samplesize(self):
        '''
        Testing that the sample size evenly divides the total number of targets
        '''
        target_list = [0,0,0,0,0,0,0,0,0]
        with self.assertRaises(ValueError):
            pr, se = get_success_rate(target_list,2,2)
        pr_cor, se_cor = get_success_rate(target_list,3,2)
        self.assertEqual([[1,0],[1,0],[1,0]], pr_cor)
        self.assertEqual([[0,0],[0,0],[0,0]], se_cor)
    
    def test_targets(self):
        '''
        Testing the values in target_list to ensure they are in the range -1,0,1,...,ntargets-1
        Note: all non integer values in the range will be counted as missing the target
        '''
        target_list1 = [-1,0.3,0,0,-1]
        target_list2 = [-1,54,23,67,1,0, 2,3,56,-1,6,4]
        pr1, se1 = get_success_rate(target_list1,5,2)
        pr2, se2 = get_success_rate(target_list2,6,3)
        expected_pr1 = [[0.4,0]]
        expected_se1 = [[np.sqrt(0.24/5),0]]
        expected_pr2 = [[1/6,1/6,0],[0,0,1/6]]
        expected_se2 = [[np.sqrt(5/216),np.sqrt(5/216),0],[0,0,np.sqrt(5/216)]]
        
        for i in range(len(expected_pr1)):
            for pr, se, ex_pr, ex_se in zip(pr1[i],se1[i],expected_pr1[i],expected_se1[i], strict=True):
                self.assertAlmostEqual(pr,ex_pr)
                self.assertAlmostEqual(se, ex_se)
        
        for i in range(len(expected_pr2)):
            for pr, se, ex_pr, ex_se in zip(pr2[i],se2[i],expected_pr2[i],expected_se2[i], strict=True):
                self.assertAlmostEqual(pr,ex_pr)
                self.assertAlmostEqual(se, ex_se)


    def test_expectedcase(self):
        '''
        Testing expected 'normal' cases
        '''
        target_list_s1 = [0, 1, -1, 0, 2, 1, 0, 0, -1, 0, 2, 1, 2, 2, 1, 0, 0, 1, 0, -1]
        target_list_s5 = [0,0,0,0,1, -1,-1,0,-1,0, 0,2,0,2,1, 1,2,1,1,0, -1,1,1,-1,-1, 1,1,1,1,1]
        target_list_s20 = [0,1,2,0,1,1,1,1,0,0,0,1,0,0,2,0,1,1,0,1, 1,-1,0,-1,-1,2,0,1,1,-1,-1,-1,0,-1,2,0,1,0,-1,1, 0,2,2,0,2,2,2,2,0,0,0,2,0,0,2,0,0,2,0,2]

        pr_s1, se_s1 = get_success_rate(target_list_s1,1,3)
        pr_s5, se_s5 = get_success_rate(target_list_s5,5,3)
        pr_s20, se_s20 = get_success_rate(target_list_s20,20,3)
        expected_pr_s1 = [[1,0,0],[0,1,0],[0,0,0],[1,0,0],[0,0,1],[0,1,0],[1,0,0],[1,0,0],[0,0,0],[1,0,0],[0,0,1],[0,1,0],[0,0,1],[0,0,1],[0,1,0],[1,0,0],[1,0,0],[0,1,0],[1,0,0],[0,0,0]]
        expected_se_s1 = [[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0]]
        expected_pr_s5 = [[4/5,1/5,0],[2/5,0,0],[2/5,1/5,2/5],[1/5,3/5,1/5],[0,2/5,0],[0,1,0]]
        expected_se_s5 = [[np.sqrt(4/125),np.sqrt(4/125),0],[np.sqrt(6/125),0,0],[np.sqrt(6/125),np.sqrt(4/125),np.sqrt(6/125)],[np.sqrt(4/125),np.sqrt(6/125),np.sqrt(4/125)],[0,np.sqrt(6/125),0],[0,0,0]]
        expected_pr_s20 = [[9/20,9/20,2/20],[5/20,5/20,2/20],[10/20,0,10/20]]
        expected_se_s20 = [[np.sqrt(99/8000),np.sqrt(99/8000),np.sqrt(36/8000)],[np.sqrt(75/8000),np.sqrt(75/8000),np.sqrt(36/8000)],[np.sqrt(100/8000),0,np.sqrt(100/8000)]]

        for i in range(len(expected_pr_s1)):
            for pr, se, ex_pr, ex_se in zip(pr_s1[i],se_s1[i],expected_pr_s1[i],expected_se_s1[i],strict=True):
                self.assertAlmostEqual(pr,ex_pr)
                self.assertAlmostEqual(se,ex_se)
        
        for i in range(len(expected_pr_s5)):
            for pr, se, ex_pr, ex_se in zip(pr_s5[i],se_s5[i],expected_pr_s5[i],expected_se_s5[i],strict=True):
                self.assertAlmostEqual(pr,ex_pr)
                self.assertAlmostEqual(se,ex_se)

        for i in range(len(expected_pr_s20)):
            for pr, se, ex_pr, ex_se in zip(pr_s20[i],se_s20[i],expected_pr_s20[i],expected_se_s20[i],strict=True):
                self.assertAlmostEqual(pr,ex_pr)
                self.assertAlmostEqual(se,ex_se)

class BumpTypeTests(unittest.TestCase):
    '''
    get_bump_type tests
    '''
    def test_dimensions():
        '''
        Testing the dimensions of the inputs: initialxt, initialyt, xPos, yPos, and activity
        checking for mismatches: initialxt and initialyt, xPos and yPos, activity and xPos (after testing to ensure xPos and yPos are the same dimension)
        '''
    
    def test_bumpneurons():
        '''
        Testing for when two expected bumps are at the same neuron
        test for when they are adjacent (and both positive, both negative, one positive one negative)
        '''
    
    def test_spanningbump():
        '''
        Testing the classification of a bump that spans all of the expected bump positions or any 2 of the bump positions, should classify it as one bump
        Include testing bumps at left and right but not center, should classify as two bumps, 
        Also test for when they are distinct bumps
        '''

    def test_bumpedgecases():
        '''
        Testing edge cases where a bump is small
        Include one neuron bumps, target neurons with activity of 0, probably good to include target neuron with activity 0 adjacent with some activity ? 
        Test for all 0
        Test for all 0 expect small bumps
        Test for all positive except one 0 neuron between    
        '''
    
    def test_proportions():
        '''
        Testing the transition from proportions into phases
        test all edge cases for the special classification 
        test when two or more proportions are equal
        '''
    
    def test_expectedcase():
        '''
        Testing a few 'normal' cases
        '''

class BifurcationAngleTests(unittest.TestCase):
    '''
    get_bifurcation_angle tests
    '''
    def test_dimensions():
        '''
        Testing to ensure the dimensions of xPos and yPos line up
        '''
    
    def test_targets():
        '''
        Testing to ensure that the targets are valid inputs and that either both are -1 or both are positive integers
        '''

    def test_targangle():
        '''
        Testing to ensure that the angle between the targets is a valid input in radians
        '''
    
    def test_anglepeaks():
        '''
        Testing edge cases for the angles between the targets, 
        many maxima/minima, no maxima/minima, 
        '''
    
    def test_unexpectedmovement():
        '''
        Testing movement that is unexpected: 
        elliptical movement and then a decision, going past the targets and then choosing one, reaching a target by making jagged decisions (ie: sinusoidal towards a target but with sharp bends)
        '''
    
    def test_validpeak():
        '''
        Testing the valid peak check, first peak is less than equal to and more than 2 degrees from start,
        one or more invalid peak and no valid peaks, one or more invalid peak and several valid peaks
        '''
    
    def test_ratio():
        '''
        Testing the calculation of the ratio
        test equal max and min, test ratio outside of expected bounds that still works (0.5 to 1), test ratio that doesn't make sense as a bifurcation point (>1), test ratio that is infeasible (greater than 2pi/angle)
        '''
    
    def test_expectedcases():
        '''
        Testing 'normal' cases
        '''
    
    

'''
metrics = unittest.TestLoader().loadTestsFromTestCase(DestinationMetricTests)
success = unittest.TestLoader().loadTestsFromTestCase(SuccessRateTests)
bump = unittest.TestLoader().loadTestsFromTestCase(BumpTypeTests)
bifurcation = unittest.TestLoader().loadTestsFromTestCase(BifurcationAngleTests)
_ = unittest.TextTestRunner().run(metrics)
_ = unittest.TextTestRunner().run(success)
_ = unittest.TextTestRunner().run(bump)
_ = unittest.TextTestRunner().run(bifurcation)
'''
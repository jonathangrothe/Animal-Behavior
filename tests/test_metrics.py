import unittest
from pathlib import Path
import numpy as np
from python_scripts.simulation_metrics import get_destination_metrics
from python_scripts.simulation_metrics import get_success_rate
from python_scripts.simulation_metrics import get_bump_type
from python_scripts.simulation_metrics import get_bifurcation_angle

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

    def test_spanningbump(self):
        '''
        Testing the classification of a bump that spans all of the expected bump positions or any 2 of the bump positions, should classify it as one bump
        Include testing bumps at left and right but not center, should classify as two bumps, 
        Also test for when they are distinct bumps
        '''
        activity = np.zeros((100,50))
        activity[0:51,:] = 1
        activity[51:,:] = -1
        activity_lc = np.zeros((100,50))
        activity_lc[20:43,:] = 1
        activity_lc[0:20] = -1
        activity_lc[43:,:] = -1
        activity_rc = np.zeros((100,50))
        activity_rc[8:31,:] = 1
        activity_rc[0:8,:] = -1
        activity_rc[31:,:] = -1
        activity_spread = np.zeros((100,50))
        activity_spread[8:18,:] = 1
        activity_spread[18:33,:] = -1
        activity_spread[33:43,:] = 1
        activity_spread[0:8,:] = -1
        activity_spread[43:,:] = -1
        activity_three = np.zeros((100,50))
        activity_three[0:11,:] = -1
        activity_three[11:15,:] = 1
        activity_three[15:22,:] = -1
        activity_three[23:28,:] = 1
        activity_three[28:35,:] = -1
        activity_three[36:40,:] = 1
        activity_three[40:,:] = -1
        #print("big bump:")
        phase_bigbump = get_bump_type(activity)
        #print("lc:")
        phase_lc = get_bump_type(activity_lc)
        #print("rc:")
        phase_rc = get_bump_type(activity_rc)
        #print("spread:")
        phase_spread = get_bump_type(activity_spread)
        #print("three:")
        phase_three = get_bump_type(activity_three)

        self.assertEqual(1,phase_bigbump)
        self.assertEqual(1,phase_lc)
        self.assertEqual(1,phase_rc)
        self.assertEqual(2,phase_spread)
        self.assertEqual(3,phase_three)
    
    def test_wrappingbumps(self):
        '''
        Testing for when the absolute values between the indices of bumps is large, and they need to wrap around the ring
        '''
        activity_2t_1b = np.zeros((100,50))
        activity_2t_1b[0:14,:] = 1
        activity_2t_1b[14:73,:] = -1
        activity_2t_1b[73:,:] = 1
        activity_2t_2b = np.zeros((100,50))
        activity_2t_2b[0:11,:] = -1
        activity_2t_2b[11:14,:] = 1
        activity_2t_2b[14:73,:] = -1
        activity_2t_2b[73:78,:] = 1
        activity_2t_2b[78:,:] = -1
        activity_3t_1b = np.zeros((100,50))
        activity_3t_2b = np.zeros((100,50))
        activity_3t_3b = np.zeros((100,50))
        activity_3t_1b[0:28,:] = 1
        activity_3t_1b[28:86,:] = -1
        activity_3t_1b[86:,:] = 1
        activity_3t_2b[0:3,:] = 1
        activity_3t_2b[3:23,:] = -1
        activity_3t_2b[23:28,:] = 1
        activity_3t_2b[28:85,:] = -1 
        activity_3t_2b[85:,:] = 1
        activity_3t_3b[0:3,:] = 1
        activity_3t_3b[3:23,:] = -1
        activity_3t_3b[23:28,:] = 1
        activity_3t_3b[28:86,:] = -1
        activity_3t_3b[86:90,:] = 1
        activity_3t_3b[90:,:] = -1

        #print("2 targets 1 bump")
        phase_2t_1b = get_bump_type(activity_2t_1b)
        #print("2 targets 2 bumps")
        phase_2t_2b = get_bump_type(activity_2t_2b)
        #print("3 targets 1 bump")
        phase_3t_1b = get_bump_type(activity_3t_1b)
        #print("3 bumps 2 targets")
        phase_3t_2b = get_bump_type(activity_3t_2b)
        #print("3 targets 3 bumps")
        phase_3t_3b = get_bump_type(activity_3t_3b)

        self.assertEqual(1,phase_2t_1b)
        self.assertEqual(2,phase_2t_2b)
        self.assertEqual(1,phase_3t_1b)
        self.assertEqual(2,phase_3t_2b)
        self.assertEqual(3,phase_3t_3b)

    def test_smallbumps(self):
        '''
        Testing edge cases where a bump is only one neuron with zeros and ones in between
        Test for all 0 expect small bumps
        Test for all positive except one 0 neuron between    
        '''
        print("smallbumps")
        activity_all0 = np.zeros((100,50))
        activity_1 = np.zeros((100,50))
        activity_1_zeros = np.zeros((100,50))
        activity_1[0,:] = 1
        activity_1[1:,:] = -1
        activity_1_zeros[33,:] = 1
        activity_2 = np.zeros((100,50))
        activity_2_zeros = np.zeros((100,50))
        activity_2[0:20,:] = -1
        activity_2[20,:] = 1
        activity_2[21,:] = -1
        activity_2[22,:] = 1
        activity_2[23:,:] = -1
        activity_2_zeros[0,:] = 1
        activity_2_zeros[57,:] = 1
        activity_3 = np.zeros((100,50))
        activity_3_zeros = np.zeros((100,50))
        activity_3[[3,20,98],:] = 1
        activity_3[np.r_[0:3,4:20,21:98,99],:] = -1
        activity_3_zeros[[6,56,61],:] = 1
        activity_many = np.zeros((100,50))
        activity_many[[10,33,35,38,50,67,68,69,90],:] = 1

        #print("all 0")
        phase_all0 = get_bump_type(activity_all0)
        #print("one 1 else -1s")
        phase_1 = get_bump_type(activity_1)
        #print("one 1 else 0")
        phase_1_0 = get_bump_type(activity_1_zeros)
        #print("two ones else -1")
        phase_2 = get_bump_type(activity_2)
        #print("two ones else 0")
        phase_2_0 = get_bump_type(activity_2_zeros)
        #print("three ones else -1")
        phase_3 = get_bump_type(activity_3)
        #print("three ones else zero")
        phase_3_zeros = get_bump_type(activity_3_zeros)
        #print("seven bumps else 0")
        phase_many = get_bump_type(activity_many)

        self.assertEqual(0,phase_all0)
        self.assertEqual(1,phase_1)
        self.assertEqual(1,phase_1_0)
        self.assertEqual(2,phase_2)
        self.assertEqual(2,phase_2_0)
        self.assertEqual(3,phase_3)
        self.assertEqual(3,phase_3_zeros)
        self.assertEqual(4,phase_many)
    
    def test_proportionsties(self):
        '''
        Testing the transition from proportions into phases
        test all edge cases for the special classification 
        test when two or more proportions are equal
        '''
        activity_tie = np.zeros((100,4))
        activity_threetie = np.zeros((100,6))
        activity_equal = np.zeros((100,10))
    
        activity_tie[[0],2:4] = 1
        activity_threetie[[0],2:4] = 1
        activity_threetie[[0,30],4:6] = 1
        activity_equal[[0],2:4] = 1
        activity_equal[[0,5],4:6] = 1
        activity_equal[[0,5,80],6:8] = 1
        activity_equal[[0,17,38,59,78],8:10] = 1

        tie = get_bump_type(activity_tie)
        threetie = get_bump_type(activity_threetie)
        equal = get_bump_type(activity_equal)

        self.assertEqual(1,tie)
        self.assertEqual(1,threetie)
        self.assertEqual(1,equal)
        
    
    def test_proportionsconversion(self):
        '''
        Testing the conversion to phase 2
        '''
        activity_converted2 = np.zeros((100,100))
        activity_not2 = np.zeros((100,100))
        activity_border2 = np.zeros((100,100))
        activity_convert_tie01 = np.zeros((100,100))
        activity_convert_tie13 = np.zeros((100,100))
        activity_convert_tie34 = np.zeros((100,100))

        activity_converted2[[14,52],0:5] = 1
        activity_converted2[0,5:] = 1
        activity_not2[[45,90],0:3] = 1
        activity_not2[46,3:] = 1
        activity_border2[[30,70],0:4] = 1
        activity_border2[45,4:] = 1
        activity_convert_tie01[[34,38],0:6] = 1
        activity_convert_tie01[19,6:53] = 1
        activity_convert_tie13[[90,97],0:10] = 1
        activity_convert_tie13[10,10:55] = 1
        activity_convert_tie13[np.r_[0:10,20:25,80:82,99],55:] = 1
        activity_convert_tie34[np.r_[29:34,78:86],0:10] = 1
        activity_convert_tie34[np.r_[0:15,17:78,90:93],10:55] = 1
        activity_convert_tie34[[0,6,21,46,89],55:] = 1

        conv_2 = get_bump_type(activity_converted2,99)
        notconv = get_bump_type(activity_not2,99)
        border = get_bump_type(activity_border2,99)
        tie_01 = get_bump_type(activity_convert_tie01,99)
        tie_13 = get_bump_type(activity_convert_tie13,99)
        tie_34 = get_bump_type(activity_convert_tie34,99)

        self.assertEqual(2,conv_2)
        self.assertEqual(1,notconv)
        self.assertEqual(2,border)
        self.assertEqual(2,tie_01)
        self.assertEqual(2,tie_13)
        self.assertEqual(2,tie_34)

    
    def test_maxtime(self):
        '''
        Testing the maxtime parameter
        '''
        activity = np.zeros((100,100))
        activity[0,0:30] = 1
        activity_limit2 = np.zeros((100,100))
        activity_limit2[[45,48],0:5] = 1
        activity_limit2[45,3:] = 1
        
        stalls = get_bump_type(activity)
        time_limit = get_bump_type(activity,50)
        notlimit = get_bump_type(activity_limit2)

        self.assertEqual(0,stalls)
        self.assertEqual(1,time_limit)
        self.assertEqual(1,notlimit)
        

    def test_expectedcase(self):
        '''
        Testing a few 'normal' cases
        '''
        activity0_path = Path('tests') / 'test_inputs' / 'bump_type' / 'activity_0bumps.csv'
        activity1_path = Path('tests') / 'test_inputs' / 'bump_type' / 'activity_1bump.csv'
        activity2_path = Path('tests') / 'test_inputs' / 'bump_type' / 'activity_2bumps.csv'
        activity3_path = Path('tests') / 'test_inputs' / 'bump_type' / 'activity_3bumps.csv'

        act0 = np.loadtxt(activity0_path, delimiter=',')
        act1 = np.loadtxt(activity1_path, delimiter=',')
        act2 = np.loadtxt(activity2_path, delimiter=',')
        act3 = np.loadtxt(activity3_path, delimiter=',')

        phase0 = get_bump_type(act0)
        self.assertEqual(0,phase0)
            
        phase1 = get_bump_type(act1)
        self.assertEqual(1,phase1)
        
        phase2 = get_bump_type(act2)
        self.assertEqual(2,phase2)
        
        phase3 = get_bump_type(act3)
        self.assertEqual(3,phase3)

class BifurcationAngleTests(unittest.TestCase):
    '''
    get_bifurcation_angle tests
    '''
    def test_dimensions(self):
        '''
        Testing to ensure the dimensions of xPos and yPos line up
        '''
        xPos = np.array([50]*10)
        yPos = np.array([50]*8)
        yPos_correct = np.array([50]*10)
        targetx = -1
        targety = -1
        targangle =2*np.pi/3
        with self.assertRaises(ValueError):
            bif_angle = get_bifurcation_angle(xPos,yPos,targetx,targety)
        correct_ind, correct_angle = get_bifurcation_angle(xPos,yPos_correct,targetx,targety)
        self.assertEqual([0,9],correct_ind)
        self.assertEqual([0],correct_angle)

    def test_targets(self):
        '''
        Testing to ensure that when the targets are both numerical a bifurcation angle is calculated, and if one of them is a string then 0 is returned
        '''
        targetx_fail = 'n'
        targety_fail = 'n'
        targetx_valid = 50-(15*np.sqrt(3))
        targety_valid = 65
        xPos = np.array([50,50,50])
        yPos = np.array([50,50,50])

        bif_ind_fail, bif_angle_fail = get_bifurcation_angle(xPos,yPos,targetx_fail,targety_fail)
        bif_ind_xfail, bif_angle_xfail = get_bifurcation_angle(xPos,yPos,targetx_fail,targety_valid)
        bif_ind_yfail, bif_angle_yfail = get_bifurcation_angle(xPos,yPos,targetx_valid,targety_fail)
       
        self.assertEqual([0],bif_ind_fail)
        self.assertEqual([0],bif_angle_fail)
        self.assertEqual([0],bif_ind_xfail)
        self.assertEqual([0],bif_angle_xfail)
        self.assertEqual([0],bif_ind_yfail)
        self.assertEqual([0],bif_angle_yfail)
        
    def test_xsmoothing(self):
        '''
        Testing the smoothing of the trajectory data
        '''
        # goal is to push function to limit for how much noise it can take
        targetx = 40
        targety = 60
        x_prebif = [50]*50
        x_postbif = np.linspace(50,40,num=31)
        y_prebif = np.linspace(0,49,num=50)
        y_postbif = np.linspace(50,60,num=31)
        xPos_nonoise = np.zeros(81)
        yPos_nonoise = np.zeros(81)
        xPos_nonoise[0:50] = x_prebif
        xPos_nonoise[50:] = x_postbif
        yPos_nonoise[0:50] = y_prebif
        yPos_nonoise[50:] = y_postbif

        xPos_spiral = np.zeros(94)
        yPos_spiral = np.zeros(94)
        xPos_spiral[0:50] = x_prebif
        yPos_spiral[0:50] = y_prebif
        xPos_spiral[50:63] = [50.14658,50.28042,50.39262,50.47733,50.53248,50.55847,50.55710,50.53088,50.48257,50.41489,50.33046,50.23171,50.12088]
        yPos_spiral[50:63] = [50-0.02182,50-0.08456,50-0.18121,50-0.30277,50-0.44035,50-0.58633,50-0.73465,50-0.88067,50-1.0210,50-1.15303,50-1.27508,50-1.38590,50-1.48464]       
        y_spiral_post_bif = np.linspace(48.4292,60,num=31)
        xPos_spiral[63:] = x_postbif
        yPos_spiral[63:] = y_spiral_post_bif

        ind_no_noise, angle_no_noise = get_bifurcation_angle(xPos_nonoise,yPos_nonoise,targetx,targety)
        ind_spiral, angle_spiral = get_bifurcation_angle(xPos_spiral,yPos_spiral,targetx,targety)
        print(f"ind: {ind_spiral}, angles: {angle_spiral}")
        self.assertEqual([0,50,80],ind_no_noise)
        self.assertAlmostEqual(3*np.pi/4,angle_no_noise[0])

    
    #def test_unexpectedmovement():
        '''
        Testing movement that is unexpected: 
        elliptical movement and then a decision, going past the targets and then choosing one, reaching a target by making jagged decisions (ie: sinusoidal towards a target but with sharp bends)
        '''
    
    #def test_validpeak():
        '''
        Testing the valid peak check, first peak is less than equal to and more than 2 degrees from start,
        one or more invalid peak and no valid peaks, one or more invalid peak and several valid peaks
        '''
    
    def test_expected60(self):
        '''
        Testing 'normal' 60 degrees between targets cases
        '''
        # add at least one or two actual sims worth of data
        # add some double bifurcations

        targetx = 50-(15*np.sqrt(3))
        targetx_reflected = 50+(15*np.sqrt(3))
        targety = 65
        xPos = np.array([50,47.5,45,42.902,targetx])
        xPos_ref = np.array([50,52.5,55,57.098,targetx_reflected])
        yPos = np.array([50,57,64,64.1,targety])
        d1_sq_1 = (xPos[2]-xPos[0])**2+(yPos[2]-yPos[0])**2
        d2_sq_1 = (xPos[4]-xPos[2])**2+(yPos[4]-yPos[2])**2
        hyp_sq_1 = (xPos[4]-xPos[0])**2+(yPos[4]-yPos[0])**2
        d1_1 = np.sqrt(d1_sq_1)
        d2_1 = np.sqrt(d2_sq_1)
        val_1 = (d1_sq_1+d2_sq_1-hyp_sq_1)/(2*d1_1*d2_1)
        angle_1 = np.arccos(val_1)

        xPos_2 = np.array([50,46.25,42.5,40.652,targetx])
        xPos_2_ref = np.array([50,53.75,57.5,59.348,targetx_reflected]) 
        yPos_2 = np.array([50,57,64,64.1,targety])
        d1_sq_2 = (xPos_2[2]-xPos_2[0])**2+(yPos_2[2]-yPos_2[0])**2
        d2_sq_2 = (xPos_2[4]-xPos_2[2])**2+(yPos_2[4]-yPos_2[2])**2
        hyp_sq_2 = (xPos_2[4]-xPos_2[0])**2+(yPos_2[4]-yPos_2[0])**2
        d1_2 = np.sqrt(d1_sq_2)
        d2_2 = np.sqrt(d2_sq_2)
        val_2 = (d1_sq_2+d2_sq_2-hyp_sq_2)/(2*d1_2*d2_2)
        angle_2 = np.arccos(val_2)

        xPos_3 = np.array([50,45,40,38.402,targetx])
        xPos_3_ref = np.array([50,55,60,61.598,targetx_reflected])
        yPos_3 = np.array([50,57,64,64.1,targety])
        d1_sq_3 = (xPos_3[2]-xPos_3[0])**2+(yPos_3[2]-yPos_3[0])**2
        d2_sq_3 = (xPos_3[4]-xPos_3[2])**2+(yPos_3[4]-yPos_3[2])**2
        hyp_sq_3 = (xPos_3[4]-xPos_3[0])**2+(yPos_3[4]-yPos_3[0])**2
        d1_3 = np.sqrt(d1_sq_3)
        d2_3 = np.sqrt(d2_sq_3)
        val_3 = (d1_sq_3+d2_sq_3-hyp_sq_3)/(2*d1_3*d2_3)
        angle_3 = np.arccos(val_3)

        xPos_4 = np.array([50,43.25,37.5,36.152,targetx])
        xPos_4_ref = np.array([50,56.75,62.5,63.848,targetx_reflected])
        yPos_4 = np.array([50,57,64,64.1,targety])
        d1_sq_4 = (xPos_4[2]-xPos_4[0])**2+(yPos_4[2]-yPos_4[0])**2
        d2_sq_4 = (xPos_4[4]-xPos_4[2])**2+(yPos_4[4]-yPos_4[2])**2
        hyp_sq_4 = (xPos_4[4]-xPos_4[0])**2+(yPos_4[4]-yPos_4[0])**2
        d1_4 = np.sqrt(d1_sq_4)
        d2_4 = np.sqrt(d2_sq_4)
        val_4 = (d1_sq_4+d2_sq_4-hyp_sq_4)/(2*d1_4*d2_4)
        angle_4 = np.arccos(val_4)

        xPos_5 = np.array([50,42.5,35,33.902,targetx])
        xPos_5_ref = np.array([50,57.5,65,66.098,targetx_reflected])
        yPos_5 = np.array([50,57,64,64.1,targety])
        d1_sq_5 = (xPos_5[2]-xPos_5[0])**2+(yPos_5[2]-yPos_5[0])**2
        d2_sq_5 = (xPos_5[4]-xPos_5[2])**2+(yPos_5[4]-yPos_5[2])**2
        hyp_sq_5 = (xPos_5[4]-xPos_5[0])**2+(yPos_5[4]-yPos_5[0])**2
        d1_5 = np.sqrt(d1_sq_5)
        d2_5 = np.sqrt(d2_sq_5)
        val_5 = (d1_sq_5+d2_sq_5-hyp_sq_5)/(2*d1_5*d2_5)
        angle_5 = np.arccos(val_5)

        print("success 1")
        bif_ind_success, bif_angle_success = get_bifurcation_angle(xPos,yPos,targetx,targety)
        bif_ind_success_ref, bif_angle_success_ref = get_bifurcation_angle(xPos_ref,yPos,targetx_reflected,targety)
        print("success2")
        bif_ind_success2, bif_angle_success2 = get_bifurcation_angle(xPos_2,yPos_2,targetx,targety)
        bif_ind_success2_ref, bif_angle_success2_ref = get_bifurcation_angle(xPos_2_ref,yPos,targetx_reflected,targety)
        print("success3")
        bif_ind_success3, bif_angle_success3 = get_bifurcation_angle(xPos_3,yPos_3,targetx,targety)
        bif_ind_success3_ref, bif_angle_success3_ref = get_bifurcation_angle(xPos_3_ref,yPos,targetx_reflected,targety)
        print("success4")
        bif_ind_success4, bif_angle_success4 = get_bifurcation_angle(xPos_4,yPos_4,targetx,targety)
        bif_ind_success4_ref, bif_angle_success4_ref = get_bifurcation_angle(xPos_4_ref,yPos,targetx_reflected,targety)
        print("success5")
        bif_ind_success5, bif_angle_success5 = get_bifurcation_angle(xPos_5,yPos_5,targetx,targety)
        bif_ind_success5_ref, bif_angle_success5_ref = get_bifurcation_angle(xPos_5_ref,yPos,targetx_reflected,targety)

        self.assertEqual([0,2,4],bif_ind_success)
        self.assertAlmostEqual(angle_1,bif_angle_success[0])
        self.assertEqual([0,2,4],bif_ind_success_ref)
        self.assertAlmostEqual(angle_1,bif_angle_success_ref[0])

        self.assertEqual([0,2,4],bif_ind_success2)
        self.assertAlmostEqual(angle_2,bif_angle_success2[0])
        self.assertEqual([0,2,4],bif_ind_success2_ref)
        self.assertAlmostEqual(angle_2,bif_angle_success2_ref[0])

        self.assertEqual([0,2,4],bif_ind_success3)
        self.assertAlmostEqual(angle_3,bif_angle_success3[0])
        self.assertEqual([0,2,4],bif_ind_success3_ref)
        self.assertAlmostEqual(angle_3,bif_angle_success3_ref[0])

        self.assertEqual([0,2,4],bif_ind_success4)
        self.assertAlmostEqual(angle_4,bif_angle_success4[0])
        self.assertEqual([0,2,4],bif_ind_success4_ref)
        self.assertAlmostEqual(angle_4,bif_angle_success4_ref[0])

        self.assertEqual([0,2,4],bif_ind_success5)
        self.assertAlmostEqual(angle_5,bif_angle_success5[0])
        self.assertEqual([0,2,4],bif_ind_success5_ref)
        self.assertAlmostEqual(angle_5,bif_angle_success5_ref[0])
    
    #def test_expected90():
        '''
        Testing 'normal' 90 degrees between targets cases
        '''

    
    #def test_expected120():
        '''
        Testing 'nomral' 120 degrees between targets cases
        '''

    
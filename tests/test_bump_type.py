'''
Bump type tests, don't really care about bump type anymore so this is not that helpful
'''
import unittest
from pathlib import Path
import numpy as np
from python_scripts.simulation_metrics import get_bump_type


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
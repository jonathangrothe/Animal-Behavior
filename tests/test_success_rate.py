'''
Success rate tests
'''
import unittest
from pathlib import Path
import numpy as np
from python_scripts.simulation_metrics import get_success_rate


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
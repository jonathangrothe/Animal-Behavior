import unittest
import numpy as np
from .. import simulation_metrics as sim_met

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
            reached, tsteps = sim_met.get_destination_metrics(xPos, yPos, targetsx, targetsy)
        targetsx2 = [40,50]
        yPos_correct = np.array([50,60,70])
        with self.assertRaises(ValueError):
            reached, tsteps = sim_met.get_destination_metrics(xPos, yPos_correct, targetsx2, targetsy)
        self.assertEqual((1,3),sim_met.get_destination_metrics(xPos, yPos_correct, targetsx, targetsy))
        
    
    def test_reaching():
        '''
        Testing the ability to detect when the agent doesn't reach a target 
        '''

    def test_samedistance():
        '''
        Testing when a target has been reached but the distance between two or more targets is the same
        '''

    def test_expectectedcase():
        '''
        Testing expected 'normal' cases
        '''

class SuccessRateTests(unittest.TestCase):
    '''
    get_succes_rate tests
    '''
    def test_samplesize():
        '''
        Testing that the sample size evenly divides the total number of targets
        '''
    
    def test_targets():
        '''
        Testing the values in target_list to ensure they are integers in the range -1,0,1,...,ntargets-1
        '''

    def test_expectedcase():
        '''
        Testing expected 'normal' cases
        '''

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
    
    


metrics = unittest.TestLoader().loadTestsFromTestCase(DestinationMetricTests)
success = unittest.TestLoader().loadTestsFromTestCase(SuccessRateTests)
bump = unittest.TestLoader().loadTestsFromTestCase(BumpTypeTests)
bifurcation = unittest.TestLoader().loadTestsFromTestCase(BifurcationAngleTests)
_ = unittest.TextTestRunner().run(metrics)
_ = unittest.TextTestRunner().run(success)
_ = unittest.TextTestRunner().run(bump)
_ = unittest.TextTestRunner().run(bifurcation)

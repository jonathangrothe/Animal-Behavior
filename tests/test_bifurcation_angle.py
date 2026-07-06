import unittest
from pathlib import Path
import numpy as np
from python_scripts.simulation_metrics import get_bifurcation_angle


class Dimensions(unittest.TestCase):
    '''
    dimensions tests
    '''
    def test_dimensions(self):
        '''
        Testing to ensure the dimensions of xPos and yPos line up
        '''
        print("Dimensions")
        xPos = np.array([50]*10)
        yPos = np.array([50]*8)
        yPos_correct = np.linspace(50,80,num=10)
        with self.assertRaises(ValueError):
            bif_angle = get_bifurcation_angle(xPos,yPos)
        correct_ind, correct_angle = get_bifurcation_angle(xPos,yPos_correct)
        self.assertEqual(2,len(correct_ind))
        self.assertEqual(1,len(correct_angle))
        self.assertEqual(0,correct_ind[0])
        self.assertAlmostEqual(np.pi,correct_angle[0])

class Thresholds(unittest.TestCase):
    '''
    Testing the threshold for what proportion of ddirection is needed to be considered above the threshold
    '''

    def test_boundary_single(self):
        '''
        Testing the boundaries for a bifurcation being above or below the threshold for a simple case of single bifurcation
        '''
        x_prebif = [50]*51
        x_postbif = np.linspace(50,40,num=31)
        y_prebif = np.linspace(0,50,num=51)
        y_postbif = np.linspace(50,60,num=31)
        xPos = np.zeros(82)
        yPos = np.zeros(82)
        xPos[0:51] = x_prebif
        xPos[51:] = x_postbif
        yPos[0:51] = y_prebif
        yPos[51:] = y_postbif
        thresholds = np.linspace(0.01,1.1,num=12)
        for item in thresholds:
            print(f"thresh: {item}")
            ind_simple, angle_simple = get_bifurcation_angle(xPos,yPos,item)
            # add tests

    
    def test_boundary_double(self):
        # add two where there is another bifurcation that is less drastic in terms of angle, 
        # one that is a direct turn and one that is a gradual turn
        x_prebif = [50]*20
        y_gradualpre = np.linspace(0,58,num=20)

        bif_range = np.linspace(np.pi/2,np.pi,num=12)
        x_gradualfirst = [50]
        y_gradualfirst = [58]
        for item in bif_range:
            x_gradualfirst.append(x_gradualfirst[-1]+1/6*np.cos(item))
            y_gradualfirst.append(y_gradualfirst[-1]+1/6*np.sin(item))
        
        y_prebif = np.linspace(0,y_gradualfirst[-1],num=20)

        x_firstbif = np.linspace(50,43,num=20)
        y_firstbif = [y_gradualfirst[-1]]*20
        x_gradual_post_first = np.linspace(x_gradualfirst[-1],43,num=20)

        bif_2_range = np.linspace(np.pi,5*np.pi/4,num=12)
        x_gradualsecond = [43]
        y_gradualsecond = [y_firstbif[-1]]
        for r in bif_2_range:
            x_gradualsecond.append(x_gradualsecond[-1]+1/12*np.cos(r))
            y_gradualsecond.append(y_gradualsecond[-1]+1/12*np.sin(r))

        x_secondbif = np.linspace(43,40,num=20)
        y_secondbif = np.linspace(y_gradualfirst[-1],53,num=20) # we need to make these angles the same

        final_heading = bif_2_range[-1]    
        seg_length = 20 * (1/12)            

        dist = np.linspace(0, seg_length, 20)
        x_post_second = x_gradualsecond[-1] + dist * np.cos(final_heading)
        y_post_second = y_gradualsecond[-1] + dist * np.sin(final_heading)

        x_immediate = np.concat([x_prebif,x_firstbif,x_secondbif])
        y_immediate = np.concat([y_prebif,y_firstbif,y_secondbif])

        
        x_gradual = np.concat([x_prebif,x_gradualfirst,x_gradual_post_first,x_gradualsecond,x_post_second])
        y_gradual = np.concat([y_gradualpre,y_gradualfirst,y_firstbif,y_gradualsecond,y_post_second])
        thresholds = np.linspace(0.05,0.7,num=15)
        for item in thresholds: 
            ind_immediate, angles_immediate = get_bifurcation_angle(x_immediate,y_immediate,item,25,5000,True)
            ind_gradual, angles_gradual = get_bifurcation_angle(x_gradual, y_gradual,item,25,5000,True)
            print(f"threshold: {item}")
            print(f"indices immediate: {ind_immediate}, angles immediate: {angles_immediate}")
            print(f"indices gradual: {ind_gradual}, angles gradual: {angles_gradual}")
            # add tests

    
class MovementThresholds(unittest.TestCase):
    '''
    Testing the threshold for beginning movement
    '''
    def test_late_movement(self): 
        '''
        Testing cases when the movement doesn't start until late, if it starts at the end then it doesn't count as movement
        '''
        xPos_static = np.linspace(50,51,num=100)
        yPos_static = np.concat([[50]*99,[80]])
        ind_static, angles_static = get_bifurcation_angle(xPos_static,yPos_static,0.25,25,5000,True)

        yPos_late = np.concat([[50]*98,[80],[80]])
        ind_late, angles_late = get_bifurcation_angle(xPos_static,yPos_late,0.25,25,5000,True)

        yPos_intime = np.concat([[50]*97,[80]*3])
        ind_intime, angles_intime = get_bifurcation_angle(xPos_static,yPos_intime,0.25,25,5000,True)

        yPos_10 = np.concat([[50]*89,[80]*11])
        ind_10, angles_10 = get_bifurcation_angle(xPos_static,yPos_10,0.25,25,5000,True)

        yPos_11 = np.concat([[50]*88,[80]*12])
        ind_11, angles_11 = get_bifurcation_angle(xPos_static,yPos_11,0.25,25,5000,True)

        yPos_11_ymove = np.concat([[50]*88,[80]*11,[85]])
        ind_11_move, angles_11_move = get_bifurcation_angle(xPos_static,yPos_11_ymove,0.25,25,5000,True)

        self.assertEqual(2,len(ind_static))
        self.assertEqual(1,len(angles_static))
        self.assertAlmostEqual(np.pi,angles_static[0])

        self.assertEqual(2,len(ind_late))
        self.assertEqual(1,len(angles_late))
        self.assertAlmostEqual(np.pi,angles_late[0])

        self.assertEqual(2,len(ind_intime)) # because its near the end it gets rejected
        self.assertEqual(0,len(angles_intime))

        self.assertEqual(2,len(ind_10))
        self.assertEqual(0,len(angles_10))

        self.assertEqual(2,len(ind_11))
        self.assertEqual(0,len(angles_11))

        self.assertEqual(3,len(ind_11_move))
        self.assertEqual(1,len(angles_11_move))
        close_to_pi = np.pi - angles_11_move[0] < 0.01
        self.assertEqual(True, close_to_pi)

    
    def test_immediate_movement(self): # ADD CASES
        '''
        Testing cases when the agent immediately moves and moves rapidly ()
        '''
        xPos_frantic = np.array([50]*100)
        yPos_frantic = np.concat([[50],[60]*9,np.linspace(60,80,num=90)])
        ind_frantic, angles_frantic = get_bifurcation_angle(xPos_frantic, yPos_frantic)
        self.assertEqual(2,len(ind_frantic))
        self.assertEqual(1,len(angles_frantic))
        self.assertAlmostEqual(np.pi,angles_frantic[0])

    def test_movement_boundaries(self):
        '''
        Testing cases when the bifurcation happens right on the border of the movement boundaries
        '''
        x_before = np.concat([np.linspace(50,50.75,num=10),np.linspace(50.75,50,num=90)])
        y_before = np.concat([np.linspace(50,50.75,num=10),np.linspace(50.75,80,num=90)])
        x_lowerbound = np.concat([[50]*10,np.linspace(50,50+15*np.sqrt(3),num=90)])
        y_lowerbound = np.concat([np.linspace(50,51.5,num=10),np.linspace(51.5,65,num=90)])
        x_post = np.concat([[50]*50,np.linspace(50,50+15*np.sqrt(3),num=50)])
        y_post = np.concat([np.linspace(50,60,num=50),np.linspace(60,65,num=50)])

        ind_before, angles_before = get_bifurcation_angle(x_before,y_before)
        ind_lowerbound, angles_lowerbound = get_bifurcation_angle(x_lowerbound,y_lowerbound)
        ind_post, angles_post = get_bifurcation_angle(x_post, y_post)

        self.assertEqual(2,len(ind_before))
        self.assertEqual(1,len(angles_before))
        self.assertAlmostEqual(np.pi,angles_before[0])
        self.assertEqual(3,len(ind_lowerbound))
        self.assertEqual(10,ind_lowerbound[1])
        self.assertEqual(1,len(angles_lowerbound))
        self.assertEqual(3,len(ind_post))
        self.assertEqual(50,ind_post[1])
        self.assertEqual(1,len(angles_post))        
        

class MaxTime(unittest.TestCase):
    '''
    Tests for the maxtime parameter
    '''
    def test_maxtime(self):
        '''
        Tests for the maxtime parameter
        '''
        print("MAXTIME")
        xPos_max = np.array([50]*5001)
        yPos_max = np.array([50]*5001)
        xPos_reaches = np.array([50]*3000)
        yPos_reaches = np.linspace(50,80,num=3000)

        ind_max, angles_max = get_bifurcation_angle(xPos_max,yPos_max)
        ind_reaches, angles_reaches = get_bifurcation_angle(xPos_reaches,yPos_reaches)
        ind_timecrunch, angles_timecrunch = get_bifurcation_angle(xPos_reaches, yPos_reaches,maxtime=2500)

        print(f"indices time crunch: {ind_timecrunch}, angles time crunch: {angles_timecrunch}")
        self.assertEqual(1,len(ind_max))
        self.assertEqual(1,len(angles_max))
        self.assertEqual(2,len(ind_reaches))
        self.assertEqual(1,len(angles_reaches))
        self.assertAlmostEqual(np.pi,angles_reaches[0])
        self.assertEqual(1,len(ind_timecrunch))
        self.assertEqual(1,len(angles_timecrunch))

class Gap(unittest.TestCase):
    def test_small_gap(self):
        '''
        Testing for when the min_gap is small 
        '''
        xPos_mini_0gap = np.concat([np.array([50]*5),[52,61,65]])
        yPos_mini_0gap = np.concat([np.linspace(50,60,num=5),[63,67,65]])
        xPos_mini_0gap_2 = np.concat([np.array([50]*5),[52,61,63,65]])
        yPos_mini_0gap_2 = np.concat([np.linspace(50,60,num=5),[63,67,66,65]])
        xPos_mini_1gap = np.concat([np.array([50]*5),[55,60,65]])
        yPos_mini_1gap = np.concat([np.linspace(50,60,num=5),[63,67,65]])

        ind_mini0, angles_mini0 = get_bifurcation_angle(xPos_mini_0gap,yPos_mini_0gap,0.25,25,5000,True)
        ind_mini0_2, angles_mini0_2 = get_bifurcation_angle(xPos_mini_0gap_2, yPos_mini_0gap_2,0.25,4,5000,True)
        ind_mini1, angles_mini1 = get_bifurcation_angle(xPos_mini_1gap,yPos_mini_1gap,0.25,25,5000,True)

        self.assertEqual(2,len(ind_mini0))
        self.assertEqual(1,len(angles_mini0))
        self.assertAlmostEqual(np.pi,angles_mini0[0])
        self.assertEqual(3,len(ind_mini0_2))
        self.assertEqual(6,ind_mini0_2[1])
        self.assertEqual(1,len(angles_mini0_2))
        self.assertEqual(3,len(ind_mini1))
        self.assertEqual(4,ind_mini1[1])

    #def test_med_gap(self):
        '''
        Testing for when the min_gap is somewhere in between 1 and 30 (max)
        '''
        #xPos_med_14gap = np.concat([])
        #yPos_med_14gap = np.concat([])
        #xPos_med_15gap = np.concat([])
        #yPos_med_15gap = np.concat([])

    #def test_large_gap(self):
        '''
        Testing for when the min_gap is 30
        '''
        #xPos_max_29gap = np.concat([])
        #yPos_max_29gap = np.concat([])
        #xPos_max_30gap = np.concat([])
        #yPos_max_30gap = np.concat([])
    

class UnexpectedMovement(unittest.TestCase):
    '''
        Testing movement that is unexpected: 
        elliptical movement and then a decision, going past the targets and then choosing one, reaching a target by making jagged decisions (ie: sinusoidal towards a target but with sharp bends)
    '''
    #def test_spiral(self):

    #def test_past(self):

    #def test_jagged(self):

    #def test_smooth(self):

class ExpectedSixty(unittest.TestCase):
    def test_expected60(self):
        '''
        Testing 'normal' 60 degrees between targets cases
        '''
        # add at least one or two actual sims worth of data
        # add some double bifurcations
        print("EXPECTED 60")
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

        bif_ind_success, bif_angle_success = get_bifurcation_angle(xPos,yPos)
        bif_ind_success_ref, bif_angle_success_ref = get_bifurcation_angle(xPos_ref,yPos)
        print(f"success 1, {bif_ind_success}, {bif_angle_success}")
        print(f"success 1 ref, {bif_ind_success_ref}, {bif_angle_success_ref}")
        print(f"x around bif: {xPos[bif_ind_success[1]-2:]}, y around bif: {yPos[bif_ind_success[1]-2:]}")

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

        bif_ind_success2, bif_angle_success2 = get_bifurcation_angle(xPos_2,yPos_2)
        bif_ind_success2_ref, bif_angle_success2_ref = get_bifurcation_angle(xPos_2_ref,yPos)
        print(f"success 2, {bif_ind_success2}, {bif_angle_success2}")
        print(f"success 2 ref, {bif_ind_success2_ref}, {bif_angle_success2_ref}")
        bif_ind_success3, bif_angle_success3 = get_bifurcation_angle(xPos_3,yPos_3)
        bif_ind_success3_ref, bif_angle_success3_ref = get_bifurcation_angle(xPos_3_ref,yPos)
        print(f"success 3, {bif_ind_success3}, {bif_angle_success3}")
        print(f"success 3 ref, {bif_ind_success3_ref}, {bif_angle_success3_ref}")
        bif_ind_success4, bif_angle_success4 = get_bifurcation_angle(xPos_4,yPos_4)
        bif_ind_success4_ref, bif_angle_success4_ref = get_bifurcation_angle(xPos_4_ref,yPos)
        print(f"success 4, {bif_ind_success4}, {bif_angle_success4}")
        print(f"success 4 ref, {bif_ind_success4_ref}, {bif_angle_success4_ref}")
        bif_ind_success5, bif_angle_success5 = get_bifurcation_angle(xPos_5,yPos_5)
        bif_ind_success5_ref, bif_angle_success5_ref = get_bifurcation_angle(xPos_5_ref,yPos)
        print(f"success 5, {bif_ind_success5}, {bif_angle_success5}")
        print(f"success 5 ref, {bif_ind_success5_ref}, {bif_angle_success5_ref}")

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
        
class ExpectedNinety(unittest.TestCase):
    #def test_expected90():
        '''
        Testing 'normal' 90 degrees between targets cases
        '''

class ExpectedOneTwenty(unittest.TestCase):
    #def test_expected120():
        '''
        Testing 'nomral' 120 degrees between targets cases
        '''

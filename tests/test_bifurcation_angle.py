import unittest
from pathlib import Path
import numpy as np
from python_scripts.simulation_metrics import get_bifurcation_angle


class BifurcationAngleTests(unittest.TestCase):
    '''
    get_bifurcation_angle tests
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
        self.assertEqual(0,correct_angle[0])

    def test_thresholds(self):
        '''
        Testing the smoothing of the trajectory data
        '''
        # honestly feeling ok about this except for the spiral case, which I will put on pause for now
        # COME BACK TO LATER TO WRITE ACTUAL TESTS
        print("TEST THRESHOLD")
        rng = np.random.default_rng(seed=67)
        
        #targetx = 40
        #targety = 60
        x_prebif = [50]*51
        x_postbif = np.linspace(50,40,num=31)
        y_prebif = np.linspace(0,50,num=51)
        y_postbif = np.linspace(50,60,num=31)
        xPos_nonoise = np.zeros(82)
        yPos_nonoise = np.zeros(82)
        xPos_nonoise[0:51] = x_prebif
        xPos_nonoise[51:] = x_postbif
        yPos_nonoise[0:51] = y_prebif
        yPos_nonoise[51:] = y_postbif

        gaussian_noise_x = rng.normal(0,0.05,82)
        gaussian_noise_y = rng.normal(0,0.05,82)
        x_withnoise = xPos_nonoise + gaussian_noise_x
        y_withnoise = yPos_nonoise + gaussian_noise_y

        spiral_period = np.linspace(0,np.pi/2,num=13)
        x_spiral = []
        y_spiral = []
        for item in spiral_period:
            x_spiral.append(50 + item*np.cos(item))
            y_spiral.append(50+ item*np.sin(item))

        spiral_period_short = np.linspace(0,np.pi/3,num=7)
        x_spiral_short = []
        y_spiral_short = []
        for item in spiral_period_short:
            x_spiral_short.append(50 + item*np.cos(item))
            y_spiral_short.append(50+ item*np.sin(item))

        spiral_period_long = np.linspace(0,np.pi,num=30)
        x_spiral_long = []
        y_spiral_long = []
        for item in spiral_period_long:
            x_spiral_long.append(50 + item*np.cos(item))
            y_spiral_long.append(50+ item*np.sin(item))

        xPos_spiral = np.zeros(95)
        yPos_spiral = np.zeros(95)
        xPos_spiral[0:51] = x_prebif
        yPos_spiral[0:51] = y_prebif
        xPos_spiral[51:64] = x_spiral
        yPos_spiral[51:64] = y_spiral     
        y_spiral_post_bif = np.linspace(48.4292,60,num=31)
        xPos_spiral[64:] = x_postbif
        yPos_spiral[64:] = y_spiral_post_bif

        xPos_spiral_short = np.zeros(89)
        yPos_spiral_short = np.zeros(89)
        xPos_spiral_short[0:51] = x_prebif
        yPos_spiral_short[0:51] = y_prebif
        xPos_spiral_short[51:58] = x_spiral_short
        yPos_spiral_short[51:58] = y_spiral_short    
        xPos_spiral_short[58:] = x_postbif
        yPos_spiral_short[58:] = y_spiral_post_bif

        xPos_spiral_long = np.zeros(112)
        yPos_spiral_long = np.zeros(112)
        xPos_spiral_long[0:51] = x_prebif
        yPos_spiral_long[0:51] = y_prebif
        xPos_spiral_long[51:81] = x_spiral_long
        yPos_spiral_long[51:81] = y_spiral_long     
        xPos_spiral_long[81:] = x_postbif
        yPos_spiral_long[81:] = y_spiral_post_bif

        thresholds = np.linspace(0.1,0.8,num=16)

        for item in thresholds: 
            #print(item)
            ind_no_noise, angle_no_noise = get_bifurcation_angle(xPos_nonoise,yPos_nonoise,item)
            ind_spiral, angle_spiral = get_bifurcation_angle(xPos_spiral,yPos_spiral,item)
            ind_spiral_short, angle_spiral_short = get_bifurcation_angle(xPos_spiral_short,yPos_spiral_short)
            #ind_spiral_long, angle_spiral_long = get_bifurcation_angle(xPos_spiral_long,yPos_spiral_long,targetx,targety)
            ind_noise, angle_noise = get_bifurcation_angle(x_withnoise,y_withnoise,item)
            #print(f"ind_no_noise: {ind_no_noise}, angle_no_noise: {angle_no_noise}")
            #print(f"x pos at indices: {xPos_nonoise[ind_no_noise]}, y pos at indices: {yPos_nonoise[ind_no_noise]}")
            #print(f"ind spiral short: {ind_spiral_short}, angles spiral short: {angle_spiral_short}")
            #print(f"x pos at indices: {xPos_spiral_short[ind_spiral_short]}, y pos at indices: {yPos_spiral_short[ind_spiral_short]}")
            #print(f"ind spiral: {ind_spiral}, angles spiral: {angle_spiral}")
            #print(f"x pos at indices: {xPos_spiral[ind_spiral]}, y pos at indices: {yPos_spiral[ind_spiral]}")
            #print(f"ind spiral long: {ind_spiral_long}, angles spiral long: {angle_spiral_long}")
            #print(f"x pos at indices: {xPos_spiral_long[ind_spiral_long]}, y pos at indices: {yPos_spiral_long[ind_spiral_long]}")
            #print(f"ind_noise: {ind_noise}, angle_noise: {angle_noise}")
            #print(f"x pos at indices: {x_withnoise[ind_noise]}, y pos at indices: {y_withnoise[ind_noise]}")
        #self.assertEqual([0,50,81],ind_no_noise)
        #self.assertAlmostEqual(3*np.pi/4,angle_no_noise[0])
        #self.assertEqual([0,50,81],ind_noise)
        #self.assertAlmostEqual(3*np.pi/4,angle_noise[0])
    
    def test_movement_threshold(self):
        '''
        Testing the identification of movement starting for a bifurcation
        Cases: 
        -movement doesn't start
        -movement starts instantly
        -movement starts at different indices in the mask
        -identifies a bifurcation before movement starts, at the border of movement starting and stopping, and after movement starts
        '''
        print("MOVEMENT THRESHOLDS")
        xPos_static = np.linspace(50,51,num=100)
        yPos_static = np.concat([[50]*99,[80]])

        xPos_frantic = np.array([50]*100)
        yPos_frantic = np.concat([[50],[60]*9,np.linspace(60,80,num=90)])

        x_before = np.concat([np.linspace(50,50.75,num=10),np.linspace(50.75,50,num=90)])
        y_before = np.concat([np.linspace(50,50.75,num=10),np.linspace(50.75,80,num=90)])

        x_lowerbound = np.concat([[50]*10,np.linspace(50,50+15*np.sqrt(3),num=90)])
        y_lowerbound = np.concat([np.linspace(50,51.5,num=10),np.linspace(51.5,65,num=90)])

        x_post = np.concat([[50]*50,np.linspace(50,50+15*np.sqrt(3),num=50)])
        y_post = np.concat([np.linspace(50,60,num=50),np.linspace(60,65,num=50)])

        ind_static, angles_static = get_bifurcation_angle(xPos_static,yPos_static,0.25,5000,True)
        ind_frantic, angles_frantic = get_bifurcation_angle(xPos_frantic, yPos_frantic)
        ind_before, angles_before = get_bifurcation_angle(x_before,y_before)
        ind_lowerbound, angles_lowerbound = get_bifurcation_angle(x_lowerbound,y_lowerbound)
        ind_post, angles_post = get_bifurcation_angle(x_post, y_post)

        print(f"indices lower bound: {ind_lowerbound}, angles lower bound: {angles_lowerbound}")
        self.assertEqual(2,len(ind_static))
        self.assertEqual(1,len(angles_static))
        self.assertEqual(0,angles_static[0])
        self.assertEqual(2,len(ind_frantic))
        self.assertEqual(1,len(angles_frantic))
        self.assertEqual(0,angles_frantic[0])
        self.assertEqual(2,len(ind_before))
        self.assertEqual(1,len(angles_before))
        self.assertEqual(0,angles_before[0])
        self.assertEqual(3,len(ind_lowerbound))
        self.assertEqual(10,ind_lowerbound[1])
        self.assertEqual(1,len(angles_lowerbound))
        self.assertEqual(3,len(ind_post))
        self.assertEqual(50,ind_post[1])
        self.assertEqual(1,len(angles_post))
    
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
        self.assertEqual(0,angles_reaches[0])
        self.assertEqual(1,len(ind_timecrunch))
        self.assertEqual(1,len(angles_timecrunch))

    
    def test_gap(self):
        '''
        Testing the minimum gap part of the detecing bif 
        cases: one above and one below for mingap of 30, mingap of 1, mingap of in between (15?)
        '''
        print("GAPS")
        xPos_mini_0gap = np.concat([np.array([50]*5),[52,61,65]])
        yPos_mini_0gap = np.concat([np.linspace(50,60,num=5),[63,67,65]])
        xPos_mini_0gap_2 = np.concat([np.array([50]*5),[52,61,63,65]])
        yPos_mini_0gap_2 = np.concat([np.linspace(50,60,num=5),[63,67,66,65]])
        xPos_mini_1gap = np.concat([np.array([50]*5),[55,60,65]])
        yPos_mini_1gap = np.concat([np.linspace(50,60,num=5),[63,67,65]])
        '''
        xPos_med_14gap = np.concat([])
        yPos_med_14gap = np.concat([])
        xPos_med_15gap = np.concat([])
        yPos_med_15gap = np.concat([])

        xPos_max_29gap = np.concat([])
        yPos_max_29gap = np.concat([])
        xPos_max_30gap = np.concat([])
        yPos_max_30gap = np.concat([])
        '''
        ind_mini0, angles_mini0 = get_bifurcation_angle(xPos_mini_0gap,yPos_mini_0gap,0.25,5000,True)
        ind_mini0_2, angles_mini0_2 = get_bifurcation_angle(xPos_mini_0gap_2, yPos_mini_0gap_2,0.25,5000,True)
        ind_mini1, angles_mini1 = get_bifurcation_angle(xPos_mini_1gap,yPos_mini_1gap,0.25,5000,True)

        print(f"ind 0: {ind_mini0}, angles 0: {angles_mini0}")
        print(f"ind 1: {ind_mini1}, angles 1: {angles_mini1}")
        print(f"ind 0 2: {ind_mini0_2}, angles 0 2: {angles_mini0_2}")

        self.assertEqual(2,len(ind_mini0))
        self.assertEqual(1,len(angles_mini0))
        self.assertEqual(0,angles_mini0[0])
        self.assertEqual(3,len(ind_mini0_2))
        self.assertEqual(4,ind_mini0_2)
        self.assertEqual(1,len(angles_mini0_2))
        self.assertEqual(3,len(ind_mini1))
        self.assertEqual(4,ind_mini1[1])


    #def test_unexpectedmovement():
        '''
        Testing movement that is unexpected: 
        elliptical movement and then a decision, going past the targets and then choosing one, reaching a target by making jagged decisions (ie: sinusoidal towards a target but with sharp bends)
        '''
    
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

        
        bif_ind_success, bif_angle_success = get_bifurcation_angle(xPos,yPos)
        bif_ind_success_ref, bif_angle_success_ref = get_bifurcation_angle(xPos_ref,yPos)
        print(f"success 1, {bif_ind_success}, {bif_angle_success}")
        print(f"success 1 ref, {bif_ind_success_ref}, {bif_angle_success_ref}")
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
    
    #def test_expected90():
        '''
        Testing 'normal' 90 degrees between targets cases
        '''

    
    #def test_expected120():
        '''
        Testing 'nomral' 120 degrees between targets cases
        '''

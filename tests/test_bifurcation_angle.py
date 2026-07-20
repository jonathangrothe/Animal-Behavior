import unittest
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from python_scripts.simulation_metrics import get_bifurcation_angle
from helper_functions.helpers import get_estimated_angle

def solve_curve_start(x0,y0, x1,y1, x2,y2, dec_len):
    # helper function for setup, solves for where the curve should start to cut corners
    # 
    a1 = np.atan2(y1-y0,x1-x0)
    a2 = np.atan2(y2-y1,x2-x1)
    aprox_angle = get_estimated_angle(a1,a2)
    dec_factor = aprox_angle*0.2/np.pi 

    if np.abs(a2 - a1) > np.pi:
        if a2 <0 and a1 >0:
            a2 += 2*np.pi
        if a2 >0 and a1 < 0:
            a2 -= 2*np.pi
    p0 = np.array([x0,y0])
    p2 = np.array([x2,y2])
    n = dec_len
    d = (a2 - a1) / (n - 1)
    scale = np.sin(n*d/2) / np.sin(d/2)
    mid_angle = a1 + (n-1)*d/2
    D = dec_factor * scale * np.array([np.cos(mid_angle),np.sin(mid_angle)])
    u1 = np.array([np.cos(a1), np.sin(a1)])
    u2 = np.array([np.cos(a2), np.sin(a2)])
    t, s = np.linalg.solve(np.column_stack([u1, -u2]), (p2 - p0) - D)
    start_points = p0 + t*u1
    finish_points = start_points + D
    
    hyp = (x2-x0)**2+(y2-y0)**2
    end_d1_sq = (finish_points[0]-x0)**2+(finish_points[1]-y0)**2
    end_d2_sq = (x2-finish_points[0])**2+(y2-finish_points[1])**2
    end_d1 = np.sqrt(end_d1_sq)
    end_d2 = np.sqrt(end_d2_sq)
    end_val = (end_d1_sq+end_d2_sq-hyp)/(2*end_d1*end_d2)
    end_angle = np.arccos(end_val)
    tolerance =  np.abs(end_angle-aprox_angle)

    return start_points[0], start_points[1], dec_factor, tolerance, aprox_angle  # (x1, y1)
    
def coords_setup(xpts,ypts,line_len,dec_len):
    # redo this 
    a1 = np.atan2(ypts[1]-ypts[0],xpts[1]-xpts[0])
    x0 = xpts[0]
    y0 = ypts[0]
    x_traj = []
    y_traj = []
    tolerances = []
    theor_angles = []
    for i in range(len(xpts)-2):
        x1 = xpts[i+1]
        y1 = ypts[i+1]
        x2 = xpts[i+2]
        y2 = ypts[i+2]
        a2 = np.atan2(y2-y1,x2-x1)
        #print(f"before update: a1: {a1}, a2: {a2}")
        if np.abs(a2-a1) > np.pi:
            #print("entered")
            if a2 < a1:
                a2 += 2*np.pi
            else:
                a2 -= 2*np.pi
        # check if dec_len is 1, if it is don't do all this
        #print(f"after update: a1: {a1}, a2: {a2}")
        if dec_len[i] > 1:
            x1_gr,y1_gr,dec_factor1, tolerance, flat_angle = solve_curve_start(x0,y0,x1,y1,x2,y2,dec_len[i])
            tolerances.append(tolerance)
            theor_angles.append(flat_angle)
            x_gr_prebif = np.linspace(x0,x1_gr,num=line_len[i])
            y_gr_prebif = np.linspace(y0,y1_gr,num=line_len[i])
            x_traj.append(x_gr_prebif)
            y_traj.append(y_gr_prebif)
            bif_range = np.linspace(a1,a2,num=dec_len[i])
            x_gradualfirst = [x1_gr]
            y_gradualfirst = [y1_gr]
            for item in bif_range:
                x_gradualfirst.append(x_gradualfirst[-1]+dec_factor1*np.cos(item))
                y_gradualfirst.append(y_gradualfirst[-1]+dec_factor1*np.sin(item))
            x_traj.append(x_gradualfirst)
            y_traj.append(y_gradualfirst)
            x0 = x_gradualfirst[-1]
            y0 = y_gradualfirst[-1]
        else:
            tolerances.append(0)
            theor_angles.append(get_estimated_angle(a1,a2))
            x_straight_through = np.linspace(x0,x1,num=line_len[i])
            y_straight_through = np.linspace(y0,y1,num=line_len[i])
            x_traj.append(x_straight_through)
            y_traj.append(y_straight_through)
            x0 = x1
            y0 = y1

        a1 = a2
        
    # go from last point to end
    final_gradual_x = np.linspace(x0,xpts[-1],line_len[-1])
    final_gradual_y = np.linspace(y0,ypts[-1],line_len[-1])
    x_traj.append(final_gradual_x)
    y_traj.append(final_gradual_y)
    final_x_traj = np.concat(x_traj)
    final_y_traj = np.concat(y_traj)
    
    total_time = len(final_x_traj)
    x_diff = np.diff(final_x_traj)
    y_diff = np.diff(final_y_traj)
    headings = np.zeros(total_time)
    headings[0] = np.atan2(ypts[1]-ypts[0],xpts[1]-xpts[0])
    for index in range(total_time-1):
        if (np.abs(x_diff[index]) < 1e-9) or (np.abs(y_diff[index]) < 1e-9):
            headings[index+1] = headings[index]
        else:
            head_angle = np.atan2(y_diff[index],x_diff[index])
            if head_angle < 0: 
                head_angle += 2*np.pi
            headings[index+1] = head_angle
    
    return [final_x_traj,final_y_traj,headings], tolerances, theor_angles

class ToleranceTesting(unittest.TestCase):
    '''
    testing some simple two bifurcation cases where the trajectory moves smoothly between three angles in a pattern of line, turn, line, turn, line
    making sure that the measured bifurcation angle is within a certain value of the theoretical bifurcation angle based on the details of the curve
    the idea is that our get_bifurcation_angle function should detect the start and end points of the turn, and either return the angle at one of them, or a 'better' angle in between
    in these simple cases the bifurcation angles at the start and end point should be the same
    '''
    def test_allquadrants(self):
        rng = np.random.default_rng()
        a1 = rng.uniform(0, np.pi / 2)
        a2 = rng.uniform(0, np.pi / 2)
        a3 = rng.uniform(0, np.pi / 2)   
        a1_list = []
        a2_list = []
        a3_list = []
        xpts_arr = np.zeros((64,4))
        ypts_arr = np.zeros((64,4))
        x0 = 50
        y0 = 50
        xpts_arr[:,0] = x0
        ypts_arr[:,0] = y0
        dist = 10
        line_lens = [30,30,30]
        dec_lens = [30,30]
        for i in range(4):
            a1_list.append(a1+(np.pi/2)*i)
            a2_list.append(a2+(np.pi/2)*i)
            a3_list.append(a3+(np.pi/2)*i)
        for ind1,item1 in enumerate(a1_list):
            for ind2,item2 in enumerate(a2_list): 
                for ind3,item3 in enumerate(a3_list):
                    x1 = x0 + dist*np.cos(item1)
                    y1 = y0 + dist*np.sin(item1)
                    x2 = x1 + dist*np.cos(item2)
                    y2 = y1 + dist*np.sin(item2)
                    x3 = x2 + dist*np.cos(item3)
                    y3 = y2 + dist*np.sin(item3)
                    row = 16*ind1+4*ind2+ind3
                    xpts_arr[row,1] = x1
                    xpts_arr[row,2] = x2
                    xpts_arr[row,3] = x3
                    ypts_arr[row,1] = y1
                    ypts_arr[row,2] = y2
                    ypts_arr[row,3] = y3
        for r in range(64):
            x = xpts_arr[r,:]
            y = ypts_arr[r,:]
            gradual_coords, tolerances, theor_angle = coords_setup(x,y,line_lens,dec_lens)
            ind, angle = get_bifurcation_angle(gradual_coords[0],gradual_coords[1],gradual_coords[2],0.01)
            print(f"theoretical angles: {theor_angle}")
            print(f"angles: {angle}")
            print(f"differences: {np.abs(np.subtract(theor_angle,angle))}")
            print(f"tolerance for first bif: {tolerances[0]}, tolerance for second bif: {tolerances[1]}")
            #plt.figure(1)
            #plt.scatter(imm_coords[0],imm_coords[1],c='blue',s=1)
            #plt.scatter(gradual_coords[0],gradual_coords[1],c='red',s=1)
            #plt.show()
            if len(angle) > 0:
                within_tol1 = np.abs(theor_angle[0]-angle[0]) <= tolerances[0]
                if not within_tol1:
                    plt.figure(1)
                    plt.scatter(gradual_coords[0],gradual_coords[1],c='red',s=1)
                    plt.show()
                self.assertEqual(True, within_tol1)
            if len(angle) > 1:
                within_tol2 = np.abs(theor_angle[1]-angle[1]) <= tolerances[1]
                print(f"WITHIN: 1: {within_tol1}, 2: {within_tol2}")
                if not within_tol2:
                    plt.figure(1)
                    plt.scatter(gradual_coords[0],gradual_coords[1],c='red',s=1)
                    plt.show()
                self.assertEqual(True, within_tol2)


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
        headings = np.array([np.pi/2]*10)
        with self.assertRaises(ValueError):
            indices, bif_angle = get_bifurcation_angle(xPos,yPos,headings)
        correct_ind, correct_angle = get_bifurcation_angle(xPos,yPos_correct,headings)
        self.assertEqual(0,len(correct_ind))
        self.assertEqual(0,len(correct_angle))

class Thresholds(unittest.TestCase):
    '''
    Testing the threshold for what proportion of ddirection is needed to be considered above the threshold
    '''

    def test_boundary_single(self):
        '''
        Testing the boundaries for a bifurcation being above or below the threshold for a simple case of single bifurcation
        '''
        xpts_list = [50,50,40]
        ypts_list = [0,50,60]
        line_lens = [50,30]
        dec_lensimm = np.array([1])
        dec_lensgr = np.array([10])
        imm_coords, tolerances, flat_angles = coords_setup(xpts_list,ypts_list,line_lens,dec_lensimm)
        gradual_coords, tolerances_gr, flat_angles_gr = coords_setup(xpts_list,ypts_list,line_lens,dec_lensgr)
        thresholds = np.linspace(0.01,1.1,num=12)
        for item in thresholds:
            ind_im, angle_im = get_bifurcation_angle(imm_coords[0],imm_coords[1],imm_coords[2],item)
            ind_gr, angle_gr = get_bifurcation_angle(gradual_coords[0],gradual_coords[1],gradual_coords[2],item)
            print(f"ind im: {ind_im}, angle_im: {angle_im}")
            print(f"ind gr: {ind_gr}, angle gr: {angle_gr}")
            if item <= 1:
                self.assertEqual(1,len(ind_im))
                self.assertEqual(1,len(ind_gr))
                self.assertEqual(1,len(angle_im))
                self.assertEqual(1,len(angle_gr))
                self.assertAlmostEqual(flat_angles[0],angle_im[0])
                within_tol = np.abs(flat_angles[0]-angle_gr[0]) <= tolerances_gr[0]
                print(f"tolerance: {tolerances_gr[0]}, actual difference: {np.abs(flat_angles[0]-angle_gr[0])}, within_tol: {within_tol}")
                self.assertEqual(True, within_tol)
            else:
                self.assertEqual(0,len(ind_im))
                self.assertAlmostEqual(0,len(ind_gr))

    
    def test_boundary_double(self):
        xpts_list = [50,80,24,34]
        ypts_list = [50,90,30,65]
        line_lens = [20,20,20]
        dec_lens = np.array([10,10])
        dec_lens_im = np.array([1,1])
        im_total = sum(line_lens)-1 
        gr_total = im_total + sum(dec_lens+1)
        imm_coords, tolerances_im, flat_angles_im = coords_setup(xpts_list,ypts_list,line_lens,dec_lens_im)
        gradual_coords, tolerances, flat_angles = coords_setup(xpts_list,ypts_list,line_lens,dec_lens)
        #plt.figure(1)
        #plt.scatter(imm_coords[0],imm_coords[1],c='blue',s=1)
        #plt.scatter(gradual_coords[0],gradual_coords[1],c='red',s=1)
        #plt.show()
        print(f'angles: {flat_angles}')
        angle_diffs = np.divide(np.subtract(np.pi,flat_angles),np.subtract(dec_lens,1))
        print(f"angle_diffs: {angle_diffs}")
        bigger_angle = max(angle_diffs)
        smaller_angle = min(angle_diffs)
        ratio = smaller_angle/bigger_angle
        thresholds = [0.05,0.1,0.4,0.49,0.5,0.51,0.6,0.7,0.8,0.9,1.1]

        for item in thresholds: 
            ind_imm, angles_imm = get_bifurcation_angle(imm_coords[0],imm_coords[1],imm_coords[2],item)
            ind_gradual, angles_gradual = get_bifurcation_angle(gradual_coords[0],gradual_coords[1],gradual_coords[2],item)
            print(f"threshold: {item}, ratio: {ratio}")
            print(f"immediate: x: {imm_coords[0][ind_imm]}, y: {imm_coords[1][ind_imm]}, angles: {angles_imm}")
            print(f"gradual: x: {gradual_coords[0][ind_gradual]}, y: {gradual_coords[1][ind_gradual]} angles: {angles_gradual}")
            if item > 1:
                self.assertEqual(0,len(ind_imm))
                self.assertEqual(0,len(ind_gradual))
                self.assertEqual(0,len(angles_imm))
                self.assertEqual(0,len(angles_gradual))
            elif ratio > item:
                self.assertEqual(2,len(ind_imm))
                self.assertEqual(2,len(ind_gradual))
                self.assertEqual(2,len(angles_imm))
                self.assertEqual(2,len(angles_gradual))
                within_tol1 = np.abs(flat_angles[0] - angles_gradual[0]) <= tolerances[0]
                within_tol2 = np.abs(flat_angles[1] - angles_gradual[1]) <= tolerances[1]
                self.assertAlmostEqual(flat_angles[0],angles_imm[0])
                self.assertAlmostEqual(flat_angles[1],angles_imm[1])
                self.assertEqual(True, within_tol1)
                self.assertEqual(True, within_tol2)
            elif ratio < item: 
                self.assertEqual(1,len(ind_imm))
                self.assertEqual(1,len(ind_gradual))
                self.assertEqual(1,len(angles_imm))
                self.assertEqual(1,len(angles_gradual))
                

class MovementThresholds(unittest.TestCase):
    '''
    Testing the threshold for beginning movement
    '''
    def test_late_movement(self):
        '''
        Testing cases when the movement doesn't start until late, if its still turning at the end then we won't count that as a bifurcation
        '''
        static_coords, static_tol, static_angles = coords_setup([50,50.99,80],[50,50,80],np.array([100,2]),np.array([1]))
        ind_static, angles_static = get_bifurcation_angle(static_coords[0],static_coords[1],static_coords[2])

        late_coords, late_tol, late_angles = coords_setup([50,50.99,80],[50,50,80],np.array([100,3]),np.array([1]))
        ind_late, angles_late = get_bifurcation_angle(late_coords[0],late_coords[1],late_coords[2])

        eleven_jump_coords, eleven_jump_tol, eleven_jump_angles = coords_setup([50,50.99,80,80],[50,50,80,85],np.array([100,12,2]),np.array([1,1]))
        ind_eleven_jump, angles_eleven_jump = get_bifurcation_angle(eleven_jump_coords[0],eleven_jump_coords[1],eleven_jump_coords[2])

        self.assertEqual(0,len(ind_static))
        self.assertEqual(0,len(angles_static))

        self.assertEqual(1,len(ind_late))
        self.assertEqual(1,len(angles_late))
        self.assertEqual(100,ind_late[0])

        self.assertEqual(1,len(ind_eleven_jump))
        self.assertEqual(1,len(angles_eleven_jump))
        self.assertEqual(100,ind_eleven_jump[0])

    
    def test_immediate_movement(self): # question - do we want to flag rapid movement and return nothing if movement is too rapid? 
        '''
        Testing cases when the agent immediately moves and moves rapidly
        '''
        frantic_coords, frantic_tol, frantic_angles = coords_setup([50,51,70],[50,60,80],np.array([1,10,100]),np.array([1]))
        ind_frantic, angles_frantic = get_bifurcation_angle(frantic_coords[0],frantic_coords[1],frantic_coords[2],0.25,5000,True)
        print(f"ind_frantic: {ind_frantic}, angles_frantic: {angles_frantic}")
        self.assertEqual(1,len(ind_frantic))
        self.assertEqual(1,len(angles_frantic))
        #self.assertAlmostEqual(np.pi,angles_frantic[0])
      
        

class MaxTime(unittest.TestCase):
    '''
    Tests for the maxtime parameter
    '''
    def test_maxtime(self): 
        '''
        Tests for the maxtime parameter
        '''
        timeout_coords, timeout_angles, timeout_angles = coords_setup([50,70],[50,70],np.array([5001]),np.array([1]))
        reach_coords, reach_tol, reach_angles = coords_setup([50,70,80],[50,70,60],np.array([3000,2000]),np.array([1]))

        ind_max, angles_max = get_bifurcation_angle(timeout_coords[0],timeout_coords[1],timeout_coords[2])
        ind_reaches, angles_reaches = get_bifurcation_angle(reach_coords[0],reach_coords[1],reach_coords[2])
        ind_timecrunch, angles_timecrunch = get_bifurcation_angle(reach_coords[0],reach_coords[1],reach_coords[2])

        self.assertEqual(0,len(ind_max))
        self.assertEqual(0,len(angles_max))
        self.assertEqual(1,len(ind_reaches))
        self.assertEqual(1,len(angles_reaches))
        self.assertEqual(0,len(ind_timecrunch))
        self.assertEqual(0,len(angles_timecrunch))

#class Gap(unittest.TestCase): # USE COORDS SETUP (if we even decide to keep this element)
    #def test_small_gap(self):
        '''
        Testing for when the min_gap is small 
        
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
        '''

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

class ExpectedSixty(unittest.TestCase): # USE COORDS SETUP
    def test_expected60(self):
        '''
        Testing 'normal' 60 degrees between targets cases
        '''
        # add at least one or two actual sims worth of data
        # add some double bifurcations
        targetx = 50-(15*np.sqrt(3))
        targetx_reflected = 50+(15*np.sqrt(3))
        targety = 65
        coords_1, tol_1, angles_1 = coords_setup([50,45,targetx],[50,64,targety],np.array([40,40]),np.array([2]))
        coords_1_gr, tol_1_gr, angles_1_gr = coords_setup([50,45,targetx],[50,64,targety],np.array([40,40]),np.array([10]))
        coords_1_ref, tol_1_ref, angles_1_ref = coords_setup([50,55,targetx_reflected],[50,64,targety],np.array([40,40]),np.array([2]))
        coords_1_ref_gr, tol_1_ref_gr, angles_1_ref_gr = coords_setup([50,55,targetx_reflected],[50,64,targety],np.array([40,40]),np.array([10]))

        ind_1, angle_1 = get_bifurcation_angle(coords_1[0],coords_1[1],coords_1[2])
        ind_1_gr, angle_1_gr = get_bifurcation_angle(coords_1_gr[0],coords_1_gr[1],coords_1_gr[2])
        ind_1_ref, angle_1_ref = get_bifurcation_angle(coords_1_ref[0],coords_1_ref[1],coords_1_ref[2])
        ind_1_ref_gr, angle_1_ref_gr = get_bifurcation_angle(coords_1_ref_gr[0],coords_1_ref_gr[1],coords_1_ref_gr[2])

        within_tol = np.abs(angles_1_gr[0]-angle_1_gr[0]) <= tol_1_gr[0]
        within_tol_ref = np.abs(angles_1_ref_gr[0]-angle_1_ref_gr[0]) <+ tol_1_ref_gr[0]

        self.assertEqual(1,len(ind_1))
        self.assertEqual(41,ind_1[0])
        self.assertEqual(1,len(angle_1))
        self.assertAlmostEqual(angles_1,angle_1)
        self.assertEqual(1,len(ind_1_gr))
        self.assertEqual(45,ind_1_gr[0])
        self.assertEqual(1,len(angle_1_gr))
        self.assertEqual(True,within_tol)
        self.assertEqual(1,len(ind_1_ref))
        self.assertEqual(41,ind_1_ref[0])
        self.assertAlmostEqual(angles_1_ref,angle_1_ref)
        self.assertEqual(1,len(angle_1_ref))
        self.assertEqual(1,len(ind_1_ref_gr))
        self.assertEqual(45,ind_1_ref_gr[0])
        self.assertEqual(1,len(angle_1_ref_gr))   
        self.assertEqual(True,within_tol_ref)     
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
